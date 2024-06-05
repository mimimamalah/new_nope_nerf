import os
import cv2
import yaml
import argparse

def load_config(config_path):
    """
    Load the configuration from a YAML file.
    
    Parameters:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image at", image_path)
        return None
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

parser = argparse.ArgumentParser(description="Process and transform images based on configuration.")
parser.add_argument("config", type=str, help="Path to configuration file.")
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)

# Check if required sections exist in the configuration
if 'dataloading' not in cfg or 'extract_images' not in cfg:
    raise ValueError("Configuration file is missing required sections.")

base_path = os.path.join(os.path.dirname(__file__), '..', cfg['dataloading']['path'])
scene_name = cfg['dataloading']['scene'][0] 
input_directory = os.path.join(base_path, scene_name, "images_before_resizing")
output_directory = os.path.join(base_path, scene_name, "images")
output_width = cfg['extract_images']['resolution'][0]
output_height = cfg['extract_images']['resolution'][1]

print("Input directory:", input_directory)
print("Output directory:", output_directory)
print("Output resolution:", output_width, "x", output_height)

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process images
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)
        resized_image = resize_image(input_image_path, output_width, output_height)
        if resized_image is not None:
            cv2.imwrite(output_image_path, resized_image)
            print(f"Resized image saved at: {output_image_path}")
        else:
            print(f"Failed to resize image: {input_image_path}")
