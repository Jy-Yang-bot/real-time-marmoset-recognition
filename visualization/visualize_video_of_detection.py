"""
Generate a video to visualize the detection results throughout the YOLO-prediced videos.
- Input: annotated video from YOLO, folder of labelings of this video, and output video name.
- Output: timeline videos of the detection labels and the annotated video.
- The trained YOLO model can detect and predict labels from one video
- Some videos may have more than 1 label class, thus it can be useful if all detected labels can be visualize as a whole

Contributor: Jiayue Yang, 2025-06-07
"""
# import libraries
import cv2
import numpy as np
import os
import re

# (to edit) define the yolo annotated video, labeling folders, and the output video directory
video_path = 'yolo_predicted_video.avi'
detections_folder = 'dir_labeling_yolo_predicted_video'
output_video = 'dir_output_vid.mp4'

# (to edit) preferred sizes and properties of the output video
output_width = 1280
video_scale = 0.6
margin_height = 30
timeline_height = 500
timeline_margin = 150
label_x = 30

# define the yolo model label classes (those may appear in the yolo annotated video)
classes = [0, 1]  # (to edit) class indices --> corresponding to label class name
class_names = {0: 'x', 1: 'y'} # (to edit) class indices : label class name
class_colors = { # (to edit) the preferred color for each class label --> can modify this to make it same as the YOLO annotation colors (optional)
    0: (255, 0, 0),  # class 0 = blue
    1: (255, 255, 0),  # class 1 = cyan
}

# load the yolo annotated video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("ERROR: could not load video")
# get the video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# put this video in thte resized output video
resized_width = int(output_width * video_scale)
resized_height = int(orig_height * resized_width / orig_width)

# create a timeline layout
timeline_width = output_width - 2 * timeline_margin
num_classes = len(classes)
row_height = timeline_height // num_classes
px_per_frame = max(1, timeline_width / total_frames)
timeline = np.ones((timeline_height, timeline_width, 3), dtype=np.uint8) * 255  # (to eedit) make the output video white background

# access the labeling folder
det_map = {}
pattern = re.compile(r'yolo_predicted_video_(\d+)\.txt') # (to edit) if the labeling .txt files have specific startings
# iterate through the labelings per frame and match them with the timeline
for fname in os.listdir(detections_folder):
    match = pattern.match(fname)
    if match:
        frame_id = int(match.group(1))
        det_map.setdefault(frame_id, []).append(
            os.path.join(detections_folder, fname)
        )

# put the labeling timeline + yolo annotated video together
for frame_idx in range(total_frames):
    x_start = int(frame_idx * px_per_frame)
    x_end = int((frame_idx + 1) * px_per_frame)
    if x_start >= timeline_width:
        break
    if x_end > timeline_width:
        x_end = timeline_width
    present = {cid: False for cid in classes}
    if frame_idx in det_map:
        for txt_path in det_map[frame_idx]:
            with open(txt_path, 'r') as f:
                for line in f:
                    cid = int(line.strip().split()[0])
                    if cid in present:
                        present[cid] = True
    # draw the color bar for each detected label class
    for i, cid in enumerate(classes):
        if present[cid]:
            y0 = i * row_height
            y1 = (i + 1) * row_height
            timeline[y0:y1, x_start:x_end] = class_colors[cid]

# write the output video
out_height = resized_height + margin_height + timeline_height
# save video
out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'), # (optional) it can be saved as .mov or .avi as well, if preferred
    fps,
    (output_width, out_height)
)

# playback loop
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0
font_scale = 1.2
thickness = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # video resizing
    frame_resized = cv2.resize(frame, (resized_width, resized_height))
    # white background
    canvas = np.ones((out_height, output_width, 3), dtype=np.uint8) * 255

    # yolo annotated video on top
    x_offset = (output_width - resized_width) // 2
    canvas[0:resized_height, x_offset:x_offset+resized_width] = frame_resized
    # timeline of the detected label classes at the bottom
    timeline_vis = timeline.copy()
    x_play = int(frame_idx * px_per_frame)
    if x_play < timeline_width:
        cv2.line(
            timeline_vis,
            (x_play, 0),
            (x_play, timeline_height),
            (0, 0, 0),  # sliding bar = a black line
            3
        )
    # ensure that the timeline is centered to the output video
    x_t_offset = timeline_margin
    canvas[resized_height+margin_height:resized_height+margin_height+timeline_height,
           x_t_offset:x_t_offset+timeline_width] = timeline_vis

    # put the label names on the far left, besides the timeline
    for i, cid in enumerate(classes):
        y = resized_height + margin_height + int((i + 0.7) * row_height)
        cv2.putText(
            canvas,
            class_names[cid],
            (label_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    out.write(canvas)
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Rendered frame {frame_idx}/{total_frames}")

# end video editing
cap.release()
out.release()
# print an ending statement
print("Video completed, the output is saved at:", output_video)

