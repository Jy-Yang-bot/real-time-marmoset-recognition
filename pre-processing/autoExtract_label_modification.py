"""
Label modification of specific marmoset identities, converting from the automatically detected face and collar.
- Input: automatic extracted labels of marmoset faces and collars (from single marmoset, automatic identity extration model).
- Output: modified names for the label classes of marmoset.
- Each input folder contains 2 label classes (face, collar) with indices (0, 1).
- As only 1 marmoset with known identity is included in the video of automatic identity extraction.
- The videos were named with marmoset identity at the beginning, thus the labeling files of the input folder start with specific marmoset name.
    - This is recommended to avoid confusion between collected videos from each marmoset from the same family, but not necessary.
- The face (0) and collar (1) index are converted to corresponding marmoset names --> for facial recognition of their family.

Example: 
If the family includes marmoset A, B, C, so in the family face recognition model the labels would be:
- marmoset A (face = 0, collar = 1)
- marmoset B (face = 2, collar = 3)
- marmoset C (face = 4, collar = 5)
So, the results of automatic facial extraction of marmoset B needs to be converted: (face 0 --> 2; collar 1 --> 3)

Contributor: Jiayue Yang, 2025-09-15
"""
import os

# directory of the txt file folder of automatic identity extraction results (1 marmoset per video per folder)
txt_folder = "automatic_extract_label" # (to edit) the YOLO output of the automatic extraction model (with marmoset faces and collars detected)

# iterate throughout the folder (i.e. frames of the input video of automatic identity extraction)
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(txt_folder, txt_file)

        # only necessary if experimenter differs videos by marmoset identity
        marm_name = "abc" # (to edit) if the frame name starts with a marmoset name
        if txt_file.startswith(marm_name): 
            # create new empty lines for label modification
            new_lines = []
            # open the file and enable editing
            with open(file_path, "r") as f:
                for line in f:
                    # each line = 1 detected bounding box/object
                    line = line.strip()
                    if not line:
                        continue

                    # separate the label index (first element of the line) from the location of the bounding box/detected object
                    parts = line.split()
                    label = parts[0]

                    # (to edit) change the label index based on the pre-defined family model index
                    # see examples: if marmoset B, face 0 --> 2, collar 1 --> 3
                    if label == "0":
                        parts[0] = "2"
                    elif label == "1":
                        parts[0] = "3"
                        
                    # ensure that the other elements of the line (i.e. location of the bounding box) remain unchanged
                    # add that to the modified label class index
                    new_lines.append(" ".join(parts))

            # update the file with the modified label index, keep the file name same
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

# print an ending statement, for the specific marmoset
print("Finished updating labels for", marm_name, "files.")

