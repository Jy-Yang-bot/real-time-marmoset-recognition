# Real-time-marmoset-recognition
This program was designed for real-time recognition of different freely moving marmosets, based on unique facial features, using the YOLOv8 algorithms.

Example applications:
- Recognizing freely moving individuals in a common marmoset family.
- Identifying animals in a real-time setting during behavioral experiments.
- Measuring the duration of animal presence within a designed experimental space.

The pipeline aims to identify common marmosets (_Callithrix jacchus_) in the laboratory captivity. We construct the pipeline based on the You Only Look Once (YOLO) algorithms, version 8 (Jocher et al., 2023).
- https://docs.ultralytics.com/models/yolov8/
- https://github.com/ultralytics/ultralytics

# Prerequisites


# Libraries Dependency
The main libraries involved in this real-time pipeline are as follows:
- pytorch
- ultralytics
- pandas
- numpy
- cv2
- matplotlib.pyplot
# Usage


# Pre-trained Model (YOLOv8)
We start the real-time marmoset recognition model with the pre-trained YOLOv8 models (YOLOv8 nano, YOLOv8 small, and YOLOv8 medium), using weights pretrained on the COCO dataset. The training of face classification model can start with the pre-trained YOLOv8 weights, with no need to start from scratch. The pre-trained YOLOv8 models involved in this pipeline are downloaded from https://docs.ultralytics.com/models/yolov8/#detection-coco (Jocher et al., 2023), while users can select the optimal pre-trained model based on image properties. 

# References
- Jocher, G., Chaurasia, A., Qiu, J., 2023. Ultralytics YOLOv8.
