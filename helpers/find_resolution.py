import cv2

def get_image_dimensions(image_path):
    """
    Get the dimensions (width, height) of an image.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        tuple: Width and height of the image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image at", image_path)
        return None
    
    # Get the dimensions of the image (height, width)
    height, width = image.shape[:2]
    return width, height

# Define image paths
image_paths = {
    'swisscube': 'data/Dark/stove/images/IMG_4571.JPG',
    'swisscube_depth': 'data/Dark/stove/dpt/IMG_4571.png',
    'ignatius': 'data/Tanks/Ignatius/images/000741.jpg',
    'ignatius_depth': 'data/Tanks/Ignatius/dpt/000741.png',
    'hubble_image': 'data/CVLab/hubble_output/images/0000.png',
    'hubble_depth': 'data/CVLab/hubble_output/dpt/0000_depth.png',
    'hubble_resized_image': 'data/CVLab/hubble_output_resized/images/0000.png',
    'hubble_resized_depth': 'data/CVLab/hubble_output_resized/dpt/0000.png',
    'hubble_resized_image_paper': 'data/CVLab/hubble_output_resized_paper/images/0000.png',
    'hubble_resized_depth_paper': 'data/CVLab/hubble_output_resized_paper/dpt/0000_shaded.png',
}

# Get and print dimensions for each image
for key, path in image_paths.items():
    width, height = get_image_dimensions(path)
    if width is not None and height is not None:
        print(f"Image width of {key}:", width)
        print(f"Image height of {key}:", height)
