import cv2
import numpy as np
import os
import re

# ================= USER SETTINGS =================
video_path = '11monMar.avi'
detections_folder = 'C:\\Users\\Alienware\\PycharmProjects\\whoTouch_perform\\labels_11monMar'
output_video = 'video_5.mp4'

output_width = 1280        # width of the output video
video_scale = 0.6          # fraction of width for the video on top
margin_height = 30         # space between video and timeline
timeline_height = 500      # total height of timeline plot
timeline_margin = 150      # horizontal margin on left and right
label_x = 30               # distance of face/collar labels from left edge of output video
# =================================================

# -------- Class definitions --------
classes = [0, 1, 2, 3]  # 0=face, 1=collar
class_names = {0: 'Young1', 1: 'collar_Young1', 2: 'Young2', 3: 'collar_Young2'}
class_colors = {
    0: (255, 0, 0),  # face = blue
    1: (255, 255, 0),  # collar = cyan
    2: (230, 165, 230),   # class 2 = white
    3: (0, 255, 0)        # class 3 = lime green

}

# ---------- Open video ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {total_frames} frames @ {fps} fps")

# ---------- Compute resized video size ----------
resized_width = int(output_width * video_scale)
resized_height = int(orig_height * resized_width / orig_width)

# ---------- Timeline layout ----------
timeline_width = output_width - 2 * timeline_margin  # smaller width
num_classes = len(classes)
row_height = timeline_height // num_classes

# Each frame maps to pixel width on the timeline
px_per_frame = max(1, timeline_width / total_frames)

timeline = np.ones((timeline_height, timeline_width, 3), dtype=np.uint8) * 255  # white background

# ---------- Index detection files ----------
det_map = {}
pattern = re.compile(r'11monMar_(\d+)\.txt')
for fname in os.listdir(detections_folder):
    match = pattern.match(fname)
    if match:
        frame_id = int(match.group(1))
        det_map.setdefault(frame_id, []).append(
            os.path.join(detections_folder, fname)
        )

print(f"Indexed {len(det_map)} detection frames")

# ---------- Build FULL timeline once ----------
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

    # Draw colored bars for each class
    for i, cid in enumerate(classes):
        if present[cid]:
            y0 = i * row_height
            y1 = (i + 1) * row_height
            timeline[y0:y1, x_start:x_end] = class_colors[cid]

# ---------- Output writer ----------
out_height = resized_height + margin_height + timeline_height
out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (output_width, out_height)
)

# ---------- Playback loop ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

font_scale = 1.2
thickness = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize video
    frame_resized = cv2.resize(frame, (resized_width, resized_height))

    # White background canvas
    canvas = np.ones((out_height, output_width, 3), dtype=np.uint8) * 255

    # Place video on top, horizontally centered
    x_offset = (output_width - resized_width) // 2
    canvas[0:resized_height, x_offset:x_offset+resized_width] = frame_resized

    # Copy timeline and draw moving playhead
    timeline_vis = timeline.copy()
    x_play = int(frame_idx * px_per_frame)
    if x_play < timeline_width:
        cv2.line(
            timeline_vis,
            (x_play, 0),
            (x_play, timeline_height),
            (0, 0, 0),  # black line
            3
        )

    # Place timeline centered under video
    x_t_offset = timeline_margin
    canvas[resized_height+margin_height:resized_height+margin_height+timeline_height,
           x_t_offset:x_t_offset+timeline_width] = timeline_vis

    # ---------- Draw labels on the left of output video ----------
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

cap.release()
out.release()
print("✅ Video saved:", output_video)
