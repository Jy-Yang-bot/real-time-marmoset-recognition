# Real-time-marmoset-recognition
This program was designed for real-time recognition of different freely moving marmosets, based on unique facial features, using the YOLOv8 algorithms.
<br />
<br />
Example applications:
- Recognizing freely moving individuals in a common marmoset family.
- Identifying animals in a real-time setting during behavioral experiments.
- Measuring the duration of animal presence within a designed experimental space.

The pipeline aims to identify common marmosets, _Callithrix jacchus_, in the laboratory captivity. We construct the pipeline based on the You Only Look Once (YOLO) algorithms, version 8 (Jocher et al., 2023).
- https://docs.ultralytics.com/models/yolov8/
- https://github.com/ultralytics/ultralytics

# Prerequisites
The hardware requirements for running ultralytics (YOLOv8) models are:
- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU with CUDA: for pyTorch
- sufficient RAM storage: 6GB is minimum and 8GB+ is preferred for better computation
- free disk space: differ by sizes of imaging dataset and sotrage required for model training. usually 20-50GB

# Libraries Dependency
The main libraries involved in this real-time pipeline are as follows:
<br />
<br />
Python:
- ultralytics: base pipeline model for object detection, segmentation, and classification
- pyTorch: deep learning networks (CNN) used for YOLO training were written in pyTorch
- pandas: reading and modifying csv file
- numpy: editing multi-dimensional array objects
- cv2: OpenCV (Open Source Computer Vision Library), image processing
- matplotlib.pyplot: plotting metrics and automatic labeling results
- json: JavaScript Object Notation, light-weighted, readable format to store detection results
- os: assessing files from computer hardware interface
- time: measuring time for the real-time operations
- collections: deque, Counter, removing and counting the elements in the detection results (of a short predefined period) <br />

MATLAB:
- imwrite: saving and processing the extracted images
- tabulate: counting label frequencies to determine the most commonly detected identity

# Usage
The ultralytic model (YOLOv8) can be installed and used in virtual environments, such as Anaconda Prompt, by running:
```markdown
pip install ultralytics
```


# Pre-trained Model (YOLOv8)
We start the real-time marmoset recognition model with the pre-trained YOLOv8 models (YOLOv8 nano, YOLOv8 small, and YOLOv8 medium), using models trained on images from the COCO dataset. The training of face classification model can start with the pre-trained YOLOv8 weights, with no need to start from scratch. The pre-trained YOLOv8 models involved in this pipeline are downloaded from https://docs.ultralytics.com/models/yolov8/#detection-coco (Jocher et al., 2023), while users can select the optimal pre-trained model based on image properties. 

# References
- Jocher, G., Chaurasia, A., Qiu, J., 2023. Ultralytics YOLOv8.
