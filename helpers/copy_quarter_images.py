import os
import shutil

# Define the source and destination directories
source_dir = "../data/CVLab/hubble_output_complete/images"
destination_dir = "../data/CVLab/hubble_output_quarter/images"
#destination_dir = "../data/CVLab/hubble_output_third/images"

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through the image files
#for i in range(0, 500, 3):
for i in range(0, 500, 4):
    # Skip the image number 201
    if i == 201:
        continue

    filename = f"{i:04}.png"
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_dir, filename)

    shutil.copy(source_path, destination_path)

print("Files copied successfully!")
