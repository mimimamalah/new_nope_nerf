import numpy as np
import json
import random
import os

# Base path setup
base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
directory = os.path.join(base_path, "CVLab/hubble_output_resized_paper")
file_path = os.path.join(directory, 'Hubble_Annotations.json')

# Read JSON data from the file
with open(file_path, 'r') as file:
    poses = json.load(file)

poses = poses[:156]
# Parameters
height = 768
width = 768
focal_length = 1

# Prepare the matrix for each pose
pose_matrices = []

for pose in poses:
    # Create the 3x4 part of the pose matrix from JSON data
    affine_matrix = np.array(pose[:-1])  # Exclude the last row which is always [0, 0, 0, 1]
    
    # Create the 1x3 vector with image properties and reshape it to (3, 1) so it matches the number of rows in affine_matrix
    image_properties = np.array([height, width, focal_length]).reshape(3, 1)
    
    # Concatenate to make a 3x5 matrix
    pose_matrix = np.hstack((affine_matrix, image_properties))
    
    # Flatten the 3x5 matrix to a 1x15 vector
    flattened_pose = pose_matrix.flatten()
    
    # Random depth bounds
    near_depth = 1
    far_depth = 500 
    
    # Concatenate the flattened pose with depth bounds
    full_vector = np.hstack((flattened_pose, [near_depth, far_depth]))
    
    # Append to list of pose matrices
    pose_matrices.append(full_vector)

# Stack all vectors to create the final Nx17 matrix
final_matrix = np.vstack(pose_matrices)

# Save to .npy file
directory = os.path.join(base_path, "CVLab/hubble_output_resized")
output_directory = os.path.join(directory, 'poses_bounds.npy')
np.save(output_directory, final_matrix)
