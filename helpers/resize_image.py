import os
import cv2

def resize_image(image_path, new_width, new_height):
    """
    Resize an image to a new width and height.
    
    Parameters:
        image_path (str): Path to the image file.
        new_width (int): New width of the image.
        new_height (int): New height of the image.
        
    Returns:
        numpy.ndarray: Resized image array.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image at", image_path)
        return None
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Directory paths
input_directory = 'data/CVLab/hubble_output/images'
output_directory = 'data/CVLab/hubble_output_resized_paper/images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# New dimensions for resizing
output_width = 768  # New width
output_height = 768  # New height

# Iterate over all image files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Path to the input and output images
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)
        
        # Resize the image
        resized_image = resize_image(input_image_path, output_width, output_height)
        
        if resized_image is not None:
            # Save the resized image
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized image saved at: {output_image_path}")
        else:
            print(f"Failed to resize image: {input_image_path}")
