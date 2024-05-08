import cv2
import numpy as np
import os

def delete_specified_files(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if "shaded" in filename or "inverted" in filename or filename.endswith(".npz"):
                os.remove(os.path.join(directory, filename))
                print(f"Deleted {filename}")

# Specify the directory from which you want to delete files
directory_to_clean = "data/CVLab/hubble_output_resized_paper/dpt/" 

# Call the function to delete specified files
delete_specified_files(directory_to_clean)

# Specify input and output directories
input_directory = "data/CVLab/hubble_output_resized_paper/dpt-anything/"
output_directory = "data/CVLab/hubble_output_resized_paper/dpt/"

def process_png_to_npz(input_dir, output_dir):
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") and "shaded" not in filename and "inverted" not in filename:
            # Load the grayscale PNG image
            #depth_map = cv2.imread(os.path.join(input_dir, filename), cv2.COLOR_BGR2GRAY)
            depth_map = cv2.imread(os.path.join(input_dir, filename))
            
            # Resize the image to 384x384
            #depth_map = cv2.resize(depth_map, (384, 384), interpolation=cv2.INTER_AREA)
            
            inverted_colored_depth_map = depth_map
            #inverted_colored_depth_map = cv2.bitwise_not(depth_map)
            # Extract the red channel
            red_channel = inverted_colored_depth_map[:, :, 2]

            # Invert the red channel so that more red corresponds to darker intensity
            inverted_red_channel = 255 - red_channel

            # Normalize the inverted red channel to [0, 1]
            normalized_inverted_red_channel = inverted_red_channel / 255.0
            
            # Create a mask to gradually decrease intensity towards the bottom
            height, width = normalized_inverted_red_channel.shape
            mask = np.zeros_like(normalized_inverted_red_channel)
            for y in range(height):
                mask[y, :] = y / height

            # Apply Gaussian blur to the mask for a small gradient effect
            blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)
            
            # Apply the blurred mask to the inverted red channel
            normalized_inverted_red_channel = normalized_inverted_red_channel * (1 - blurred_mask)

            # Darken the normalized inverted red channel (multiply by a factor less than 1)
            darkened_normalized_inverted_red_channel = normalized_inverted_red_channel * 0.75  # Adjust the factor as needed
            
            # Define parameters for the sigmoid function
            center = -0.5  # Center of the sigmoid function shifted to the bottom
            steepness = 15  # Steepness of the sigmoid function

            # Create a sigmoid function to modulate the gradient strength
            sigmoid_gradient = 1 / (1 + np.exp(-steepness * (np.linspace(0, 1, darkened_normalized_inverted_red_channel.shape[0]) - center)))

            # Reshape the sigmoid gradient to match the image dimensions
            sigmoid_gradient = sigmoid_gradient.reshape(-1, 1)        

            # Apply the sigmoid gradient to the darkened image
            soft_gradient_darkened_image = darkened_normalized_inverted_red_channel * sigmoid_gradient
            #soft_gradient_darkened_image = darkened_normalized_inverted_red_channel * darkened_normalized_inverted_red_channel
 
            # Convert the darkened normalized inverted red channel to grayscale
            depth_map = (soft_gradient_darkened_image * 255).astype(np.uint8)
            #depth_map = soft_gradient_darkened_image                        
            #depth_map = cv2.bitwise_not(depth_map)
            """
            # Normalize the input image to the range [0, 1]
            normalized_input_image = depth_map / 255.0

            # Create a vertical gradient with the bottom darker than the top
            height = normalized_input_image.shape[0]
            vertical_gradient = np.linspace(1, 0, height).reshape(-1, 1)
            vertical_gradient = np.repeat(vertical_gradient, normalized_input_image.shape[1], axis=1)

            # Apply the gradient to the image
            gradient_image = normalized_input_image * vertical_gradient

            # Invert the gradient image to get a darker bottom and lighter top
            inverted_gradient_image = 1 - gradient_image

            # Enhance contrast by ensuring we have absolute black and absolute white
            # Set the minimum value of the gradient image to black and the maximum to white
            min_val, max_val = inverted_gradient_image.min(), inverted_gradient_image.max()
            contrast_enhanced_image = (inverted_gradient_image - min_val) / (max_val - min_val)

            # Rescale to 8-bit and prepare for saving
            final_image = np.clip(contrast_enhanced_image * 255, 0, 255).astype(np.uint8)
            """
            # Get the filename without extension
            file_prefix = filename.split("_")[0]
            
            depth_map_shape = depth_map.shape  # Get the shape of depth_map for debugging
            print(f"Depth map shape before saving: {depth_map_shape}")  # Print shape for debugging
            # Save the shaded image as NPZ with key 'pred'
            np.savez(os.path.join(output_dir, f"depth_{file_prefix}.npz"), pred=depth_map)
            # Save the shaded image as PNG
            cv2.imwrite(os.path.join(output_dir, f"{file_prefix}_shaded.png"), depth_map)

            print(f"Processed {filename}")

# Call the function to process PNG files in the input directory and save as NPZ in the output directory
process_png_to_npz(input_directory, output_directory)
