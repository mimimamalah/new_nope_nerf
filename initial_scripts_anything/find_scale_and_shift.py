import numpy as np
import os
import imageio
import torch
import torch.nn.functional as F

"""
This script calculates the appropriate scale and shift parameters to adapt the depth values 
from DPT models for compatibility with the NOPE-NeRF model requirements.
"""


# Function to find scale and shift parameters
def find_scale_and_shift(tensor1, tensor2):
    A = np.vstack([tensor1, np.ones(len(tensor1))]).T
    m, c = np.linalg.lstsq(A, tensor2, rcond=None)[0]
    return m, c

# Function to apply scale and shift to a tensor
def apply_scale_and_shift(tensor, scale, shift):
    transformed_tensor = tensor * scale + shift
    return transformed_tensor

# Function to apply the inverse depth transformation
def inverse_depth_transformation(depth):
    # Apply scale and shift
    depth = 0.000305 * depth + 0.1378
    
    # Apply the inverse depth transformation
    depth[depth < 1e-8] = 1e-8
    depth = 1.0 / depth
    
    return depth

depth_anything_dir = "data/KITTI/straight-anything-scale-original/dpt-anything"
depth_original_dir = "data/KITTI/straight/dpt-before"
transformed_dir = "depth_anything_scripts/KITTI-final-between-dpt-inverse-anything"

if not os.path.exists(transformed_dir):
    os.makedirs(transformed_dir)

# Set up directories
"""
depth_anything_dir = "data/CVLab/hubble_output_resized_paper/dpt-to-use"
depth_original_dir = "data/CVLab/hubble_output_resized_paper/dpt-before-inversing"
transformed_dir = "data/CVLab/hubble_output_resized_paper/transformed"
depth_save_dir = "data/CVLab/hubble_output_resized_paper/transformed"
additional_transformed_dir = "data/CVLab/hubble_output_resized_paper/dpt"
"""

# Threshold for background values (values close to zero)
background_threshold = 2

# Initialize arrays to store all non-background values
all_depth_anything_values = []
all_depth_original_values = []

# Count the number of background values removed
removed_background_count = 0

# Get list of files in directories
depth_anything_files = os.listdir(depth_anything_dir)
depth_original_files = os.listdir(depth_original_dir)

# Iterate over files to collect non-background values
for file_name in depth_anything_files:
    # Check if corresponding file exists in depth-original directory
    if file_name in depth_original_files:
        # Load tensors
        if not file_name.endswith('.npz'):
            continue
        depth_anything_path = os.path.join(depth_anything_dir, file_name)
        depth_original_path = os.path.join(depth_original_dir, file_name)
        depth_anything = np.squeeze(np.load(depth_anything_path)['pred'])
        depth_original = np.squeeze(np.load(depth_original_path)['pred'])
        
        if depth_anything.shape != depth_original.shape:
            print(f"Shapes do not match for {file_name}")
            depth_anything = torch.tensor(depth_anything, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            depth_original = torch.tensor(depth_original, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            depth_anything = F.interpolate(depth_anything, size=(depth_original.shape[2], depth_original.shape[3]), mode='bilinear', align_corners=False)
            # Remove the added dimensions
            depth_anything = depth_anything.squeeze()
            depth_original = depth_original.squeeze()
            
            #print("depth anything shape: ", depth_anything.shape)
            #print("depth original shape: ", depth_original.shape)
            
            # Identify background values in depth-anything tensor
        background_mask = np.isclose(depth_anything, 0, atol=background_threshold)

        # Count number of background values removed
        removed_background_count += np.sum(background_mask)

        # Collect non-background values from depth-anything and depth-original tensors
        #all_depth_anything_values.extend(depth_anything[~background_mask].flatten())
        #all_depth_original_values.extend(depth_original[~background_mask].flatten())
        all_depth_anything_values.extend(depth_anything.flatten())
        all_depth_original_values.extend(depth_original.flatten())

# Convert lists to numpy arrays
all_depth_anything_values = np.array(all_depth_anything_values)
all_depth_original_values = np.array(all_depth_original_values)

# Find scale and shift parameters based on all non-background values
scale, shift = find_scale_and_shift(all_depth_anything_values, all_depth_original_values)

# Save scale and shift parameters to a text file
with open(os.path.join(transformed_dir, 'scale_and_shift.txt'), 'w') as f:
    f.write(f"Global Scale: {scale}\n")
    f.write(f"Global Shift: {shift}\n")

# Print the number of background values removed
print(f"Number of background values removed: {removed_background_count}")
"""
# Iterate over files to apply the global scale and shift and save the transformed tensors
for file_name in depth_anything_files:
    # Check if corresponding file exists in depth-original directory
    if file_name in depth_original_files:
        # Load tensors
        depth_anything_path = os.path.join(depth_anything_dir, file_name)
        depth_original_path = os.path.join(depth_original_dir, file_name)
        depth_anything_shape = np.load(depth_anything_path)['pred'].shape 
        depth_anything = np.squeeze(np.load(depth_anything_path)['pred'])

        # Apply transformation to depth-anything tensor
        transformed_depth_anything = apply_scale_and_shift(depth_anything, scale, shift)

        # Save image
        depth_array = transformed_depth_anything
        converted_depths = depth_array.reshape(depth_anything_shape)
        depth_array = converted_depths[0] 
        imageio.imwrite(os.path.join(
            depth_save_dir, 
            '{}.png'.format(file_name.split('.')[0])), 
            np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8))

        # Save transformed tensor
        if not os.path.exists(transformed_dir):
            os.makedirs(transformed_dir)
        transformed_file_path = os.path.join(transformed_dir, file_name)
        np.savez(transformed_file_path, pred=converted_depths)
        
        if not os.path.exists(additional_transformed_dir):
            os.makedirs(additional_transformed_dir)
        
        additional_transformed_depth = inverse_depth_transformation(transformed_depth_anything)
        additional_transformed_file_path = os.path.join(additional_transformed_dir, file_name)
        
        depth_array = additional_transformed_depth 
        converted_depths = depth_array.reshape(depth_anything_shape)
        depth_array = converted_depths[0]
        
        np.savez(additional_transformed_file_path, pred=converted_depths)
        
        imageio.imwrite(os.path.join(
            additional_transformed_dir, 
            '{}.png'.format(file_name.split('.')[0])), 
            np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8))
"""