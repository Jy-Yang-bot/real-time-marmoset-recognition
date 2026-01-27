"""
Real-time YOLOv8 detection of which marmoset is presented for touchscreen task.
- Each second, the most common detection results from past 30 frames saved.
- Updating JSON file.
"""
from ultralytics import YOLO
import cv2
import json
from collections import deque
import os
import time
from collections import Counter


# load pretrained model
model = YOLO('C:\\Users\\jyang291\\PyCharmMiscProject\\best.pt')  # change to customized model path

# open camera
cap = cv2.VideoCapture(0)  # 0 for default camera

# customized labels
class_names = {0: 'Simon', 1: 'Amelia', 2: 'Dale', 3: 'green', 4: 'pink', 5: 'orange'}

# output file path
true_output = "C:\\MonkeyLogicInstallation\\Add-Ons\\Apps\\NIMHMonkeyLogic22\\whoTouch_detections.json"
temp_output = "C:\\MonkeyLogicInstallation\\Add-Ons\\Apps\\NIMHMonkeyLogic22\\whoTouch_detections_temp.json"

# store all detection results
all_detect = []

# write json and deal with random error on same time access of files
def try_write_json():
    attempt = 0
    max_attempts = 10
    while attempt < max_attempts:
        try:
            # Try to write the JSON file
            with open(temp_output, 'w') as f:
                json.dump({'detections': all_detect}, f)

            # replace original file with temp file
            if os.path.exists(temp_output):
                os.replace(temp_output, true_output)
            print("File written successfully.")
            return True  # success
        except PermissionError:
            # If file is locked, increment attempt, wait and try again
            print(f"Attempt {attempt + 1} failed: File is locked. Retrying...")
            attempt += 1
            time.sleep(0.2)  # wait 200 ms before trying again
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Error during file write: {e}")
            return False  # failed

    print("Max attempts reached. Could not write to file.")
    return False  # failed after max attempts

while cap.isOpened():
    # read 1 frame
    ret, frame = cap.read()
    if not ret:
        break

    # use pretrained model to predict
    results = model(frame)

    for result in results:
        boxes = result.boxes  # retrieve detection rectangle
        for box in boxes:
            # retrieve ID
            cls_id = int(box.cls[0])
            # retrieve label type
            label = class_names.get(cls_id, 'unknown')
            # skip if unknown detection or no detection
            if label == 'unknown':
                continue
            # replace with weighted name if bead colors
            if label == 'green':
                all_detect.extend(['Simon']*10)
            elif label == 'pink':
                all_detect.extend(['Amelia']*10)
            elif label == 'orange':
                all_detect.extend(['Dale']*10)
            else:
                all_detect.append(label)

    # attempt to save weighted detections in JSON file
    if not try_write_json():
        print("Failed to write detections to file. Continuing to next frame.")
    # display results per frame（optional）
    #cv2.imshow('YOLOv8 Real-time Detection', frame)

    # press 'q' hotkey to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resource
cap.release()
cv2.destroyAllWindows()