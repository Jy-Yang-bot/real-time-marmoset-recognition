"""
import os

txt_folder = "C:\\Users\\jiayue\\Desktop\\labels"     # folder with .txt labels

total_files = 0

for filename in os.listdir(txt_folder):
    if not filename.endswith(".txt"):
        continue

    total_files += 1
    txt_path = os.path.join(txt_folder, filename)

    # Read original lines
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    new_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue  # skip malformed lines

        category = int(parts[0])

        # 0,1,2  → 0
        if category in [0, 1, 2]:
            new_lines.append("0 " + " ".join(parts[1:]))

        # 3,4,5  → 1
        elif category in [3, 4, 5]:
            new_lines.append("1 " + " ".join(parts[1:]))

    # Overwrite file
    with open(txt_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"Processed: {filename}")

print("\nDone! Total txt files processed:", total_files)
"""
import os

# Path to the folder containing your TXT files
txt_folder = "C:\\Users\\jiayue\\Desktop\\labels"

# Loop over all txt files
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(txt_folder, txt_file)

        # Only process files starting with 'mim'
        if txt_file.startswith("mim"):
            new_lines = []
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    label = parts[0]

                    # Change label based on original category
                    if label == "0":
                        parts[0] = "2"
                    elif label == "1":
                        parts[0] = "3"
                    # Keep other labels unchanged if needed

                    new_lines.append(" ".join(parts))

            # Overwrite the original file with updated labels
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

print("Finished updating labels for 'mim' files.")
