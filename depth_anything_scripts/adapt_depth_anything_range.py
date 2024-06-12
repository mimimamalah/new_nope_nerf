import os
import argparse
import yaml
import numpy as np
import imageio

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

def write_scale_and_shift(file_path, scale, shift):
    with open(file_path, 'w') as f:
        f.write(f"Scale: {scale}\n")
        f.write(f"Shift: {shift}\n")

def apply_scale_and_shift(tensor, scale, shift):
    return tensor * scale + shift

def inverse_depth_transformation(depth):
    # Apply the inverse depth transformation
    depth[depth < 1e-8] = 1e-8
    transformed_depth = 1.0 / depth
    return transformed_depth

def normalize_depth(depth, min_depth, max_depth):
    return (depth - min_depth) / (max_depth - min_depth)

def scale_and_shift_normalized_depth(normalized_depth, target_min, target_max):
    scale = target_max - target_min
    shift = target_min
    return normalized_depth * scale + shift

parser = argparse.ArgumentParser(description="Process and transform images based on configuration.")
parser.add_argument("config", type=str, help="Path to configuration file.")
args = parser.parse_args()

cfg = load_config(args.config)

if 'dataloading' not in cfg:
    raise ValueError("Configuration file is missing the 'dataloading' section.")

base_path = os.path.join(os.path.dirname(__file__), '..', cfg['dataloading']['path'])
scene_name = cfg['dataloading']['scene'][0]

input_dir = os.path.join(base_path, scene_name, "dpt-before")
output_dir = os.path.join(base_path, scene_name, "dpt-range")

os.makedirs(output_dir, exist_ok=True)

base_path = os.path.join(os.path.dirname(__file__), '..', 'data')

scale_shift_file = os.path.join(os.path.dirname(__file__), 'scale_and_shift.txt')

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    if file_path.endswith('.npz'):
        output_path = os.path.join(output_dir, file_name)
        output_image_path = os.path.join(output_dir, f"{file_name.split('.')[0]}.png")

        depth_data = np.squeeze(np.load(file_path)['pred'])

        print(f"Depth data shape: {depth_data.shape}, min: {depth_data.min()}, max: {depth_data.max()}")

        # Normalize the depth data
        min_depth = depth_data.min()
        max_depth = depth_data.max()
        normalized_depth = normalize_depth(depth_data, min_depth, max_depth)

        # Scale and shift the normalized depth data to the desired range [0.1, 100]
        target_min = 0.1
        target_max = 100
        scaled_depth = scale_and_shift_normalized_depth(normalized_depth, target_min, target_max)

        # Calculate the scale and shift used
        scale = target_max - target_min
        shift = target_min
        write_scale_and_shift(scale_shift_file, scale, shift)
        print(f"Calculated Scale: {scale}, Shift: {shift}")

        print(f"After scaling and shifting, min: {scaled_depth.min()}, max: {scaled_depth.max()}")

        # Apply inverse depth transformation
        transformed_depth = inverse_depth_transformation(scaled_depth)
        print(f"After inverse transformation, min: {transformed_depth.min()}, max: {transformed_depth.max()}")

        np.savez(output_path, pred=transformed_depth)

        image = np.clip(255.0 / transformed_depth.max() * (transformed_depth - transformed_depth.min()), 0, 255).astype(np.uint8)
        imageio.imwrite(output_image_path, image)

        print(f"Processed and saved: {file_name}")
