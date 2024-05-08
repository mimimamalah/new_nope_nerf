import cv2
import numpy as np
import os

# Load the intrinsic matrix from the file
intrinsics_file = np.load('data/CVLab/SwissCube/intrinsics.npz')
K = intrinsics_file['K']

# Directory containing the original images
image_dir = 'data/CVLab/SwissCube/images'

# Directory to save cropped images
output_dir = 'data/CVLab/SwissCube_cropped/images'
os.makedirs(output_dir, exist_ok=True)

# List all files in the directory
image_files = [f for f in os.listdir(image_dir) if not f.startswith('.')]

sample_filename = image_files[0]

# Load the image using OpenCV
sample_image = cv2.imread(os.path.join(image_dir, sample_filename))

# Get the width of the image
original_width = sample_image.shape[1]
original_height = sample_image.shape[0]

# Desired dimensions of the cropped region
cropped_width = int(sample_image.shape[1] * 2/3)
cropped_height = 768

# Calculate the top-left corner coordinates for cropping
x = original_width - cropped_width  # Starting from the right edge
y = original_height - cropped_height  # Starting from the bottom edge

print("Original width:", original_width)
print("Original height:", original_height)

# Iterate over all images in the directory
for filename in image_files:
    # Generate the file path
    input_filepath = os.path.join(image_dir, filename)

    # Load the original image
    image = cv2.imread(input_filepath)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to read image '{filename}'")
        continue

    # Crop the image
    cropped_image = image[int(y):, int(x):]

    # Save the cropped image to the output directory
    output_filepath = os.path.join(output_dir, filename)
    cv2.imwrite(output_filepath, cropped_image)


# Adjust the intrinsics for cropping
K[0, 2] -= x  # Adjust cx
K[1, 2] -= y  # Adjust cy

# Save the adjusted intrinsics to a new file in the output directory
adjusted_intrinsics_file = os.path.join('data/CVLab/SwissCube_cropped', 'intrinsics.npz')
np.savez(adjusted_intrinsics_file, K=K)

print("Cropping and adjustment completed for all images.")
