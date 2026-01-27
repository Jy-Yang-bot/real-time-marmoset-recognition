"""
Real-time YOLOv8 detection of which marmoset is presented for touchscreen task.
- Input: real-time footage from USB camera, trained weights of the subject marmoset.
- Output: automated detection results of recognized subject animal.
- Each second, the most common detection results from past 30 frames saved.
- Updating JSON file.

Contributor: Jiayue Yang, 2025-03-18
"""
from ultralytics import YOLO
import cv2
import json
from collections import deque
import os
import time
from collections import Counter


# load pretrained model
model = YOLO('trained_weight_file.pt')  # (to edit) change to customized model path

# open camera
cap = cv2.VideoCapture(0)  # 0 for default camera (if othre cameras connecting to the computer, modify this)

# customized labels
class_names = {0: 'x', 1: 'y'} # (to edit) the label index and the class name of the pre-trained weight file (e.g. label index:'label name')

# output file path
true_output = "true_animal_detection_results.json" # (to edit) true output path of the JSON file
temp_output = "temporary_animal_detection_results.json" # (to edit) temporary output path of the JSON file (before saving to true file)

# store all detection results
all_detect = []

# write json and deal with random error on same time access of files
def try_write_json():
    attempt = 0
    max_attempts = 10
    while attempt < max_attempts:
        try:
            # try to write the JSON file
            with open(temp_output, 'w') as f:
                json.dump({'detections': all_detect}, f)

            # replace original file with temp file
            if os.path.exists(temp_output):
                os.replace(temp_output, true_output)
            print("File written successfully.")
            return True  # success
        # warning or error code if cannot access the file or the file is currently in use/read
        except PermissionError:
            # if file is locked, pause reading attempt, wait and try again
            print(f"Attempt {attempt + 1} failed: File is locked. Retrying...")
            attempt += 1
            time.sleep(0.2)  # wait 200 ms before reading the file again
        except Exception as e:
            # if other unexpected errors happen, report
            print(f"Error during file write: {e}")
            return False  # failed

    print("Max attempts reached. Could not write to file.")
    return False  # failed after max attempts

# real-time recognition processing
while cap.isOpened():
    # read 1 frame for prediction (for the real-time recognition)
    ret, frame = cap.read()
    if not ret:
        break

    # use pretrained model to predict
    results = model(frame)

    # based on the detection result
    for result in results:
        boxes = result.boxes  # retrieve detection rectangle
        for box in boxes:
            # retrieve label ID (which is indicative of the label name)
            cls_id = int(box.cls[0])
            # retrieve label type
            label = class_names.get(cls_id, 'unknown')
            # skip if unknown detection or no detection
            if label == 'unknown':
                continue
            # replace with weighted name if marmoset bead colors detected, otherwise append the detected subject name
            # weighted detection --> if xx color belongs to x individual
            # (to edit) bead color-subject name weight can be adjusted
            if label == 'xx':
                all_detect.extend(['x']*5)
            elif label == 'yy':
                all_detect.extend(['y']*5)
            else:
                all_detect.append(label)

    # attempt to save weighted detections in JSON file
    if not try_write_json():
        print("Failed to write detections to file. Continuing to next frame.")
    # display results per frame（optional）
    #cv2.imshow('YOLOv8 Real-time Detection', frame)

    # press 'q' hotkey to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resource
cap.release()

cv2.destroyAllWindows()
