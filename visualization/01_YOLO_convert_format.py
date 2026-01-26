## YOLO format conversion
# This program converts the text file labelings in YOLO into the csv file.
# Each txt file denotes the object detection in 1 frame will be transferred to 1 row of labeling in the csv file.
# Contributor: Jiayue Yang, 2024-09-05

# Update:
# 1. Repeat conversion for the whole length of video, 2024-09-06

# import libraries
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


# labeling index to name
#label2name = {0: 'Aubergine', 1: 'Asperge', 2: 'Artichaut'}
label2name = {0: 'touch', 1: 'hand'}

# initiate frame number counting from 1
frame = 1

# define output csv path
csv_path = 'C:\\Users\\jyang291\\Desktop\\concat_labels.csv'

"""
Step 1: Extract the total frame number, width, and height of the detection video
"""

# detection video path
#vid_path = "C:\\Users\\jyang291\\Desktop\\bird_cage\\cropped.mp4"
vid_path = "C:\\Users\\jyang291\\Desktop\\hand_touch\\IMG_0607.avi"

# read video
vid = cv2.VideoCapture(vid_path)

# find total frame number
length = int(vid. get(cv2. CAP_PROP_FRAME_COUNT))

# find the width and height of the video
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


"""
### Step 2: Read the labeling txt file and perform conversion
"""
# create a dataframe to save the results
conversion = []

# iterate through each frame
while frame <= length:
    try:
        # load the txt file
        file_path = 'C:\\Users\\jyang291\\Desktop\\hand_touch\\labels\\IMG_0607_%s.txt' % frame

        # read the file as dataframe
        df = pd.read_csv(file_path, delimiter='\t', header=None)


        # iterate through each row
        for index, row in df.iterrows():
            position = row.tolist()
            # split the elements by blank
            split_elements = position[0].split(' ')
            # convert the ratio to pixels
            x_convert = (float(split_elements[1]) - (float(split_elements[3]) / 2)) * width
            y_convert = (float(split_elements[2]) - (float(split_elements[4]) / 2)) * height
            width_convert = float(split_elements[3]) * width
            height_convert = float(split_elements[4]) * width
            # convert the middle point of the rectangle
            X_mid = (float(split_elements[1])) * width
            Y_mid = (float(split_elements[2])) * height
            # same a new list
            name_label = int(split_elements[0])
            temp_convert = [frame, label2name[name_label], x_convert, y_convert, width_convert, height_convert, X_mid, Y_mid]

            # append the results
            conversion.append(temp_convert)

        # increase the frame number
        frame += 1

    except FileNotFoundError:
        # Skip to the next iteration if the file is not found
        frame += 1
        continue

print("Conversion completed for length %s frames." % (frame - 1))



### Step 3: Save the converted tracking information in a csv file
"""
"""
# define the column names
column_names = ['Time', 'Name', 'X', 'Y', 'Width', 'Height', 'X_mid', 'Y_mid']

# save the converted data in a dataframe
py_format = pd.DataFrame(conversion)
py_format.columns = column_names

# save the dataframe into a csv file
py_format.to_csv(csv_path, index=False)

print("Labeling csv file saved.")



"""
### Step 4: Draw the rectangle on the image to confirm

# display the image and the rectangle drawn
x = np.array(Image.open(image_path), dtype=np.uint8)
plt.imshow(x)

# create figure and axes
fig, ax = plt.subplots(1)

# display the image
ax.imshow(x)

# create a Rectangle patch
rect = patches.Rectangle((py_format.loc[2, 'X'], py_format.loc[2, 'Y']),
                         py_format.loc[2, 'Width'], py_format.loc[2, 'Height'],
                         linewidth=2, edgecolor='r', facecolor="none")

# Add the patch to the Axes
ax.add_patch(rect)
plt.show()
"""



