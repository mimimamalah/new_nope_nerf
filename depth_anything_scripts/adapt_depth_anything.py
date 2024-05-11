import os
import numpy as np
import imageio

base_path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Define paths relative to the base path
input_dir = os.path.join(base_path, "CVLab/hubble_output_complete/dpt-anything")
output_dir = os.path.join(base_path, "CVLab/hubble_output_complete/dpt")

os.makedirs(output_dir, exist_ok=True)

def read_scale_and_shift(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        scale = float(lines[0].split(': ')[1].strip())
        shift = float(lines[1].split(': ')[1].strip())
    return scale, shift

def apply_scale_and_shift(tensor, scale, shift):
    return tensor * scale + shift

def inverse_depth_transformation(depth):
    # Apply in addition the same scale and shift then for the original dpt model
    depth = 0.000305 * depth + 0.1378
    
    # Apply the inverse depth transformation
    depth[depth < 1e-8] = 1e-8
    transformed_depth = 1.0 / depth
    return transformed_depth
    
scale_shift_file = os.path.join(os.path.dirname(__file__), 'scale_and_shift.txt')
scale, shift = read_scale_and_shift(scale_shift_file)
print(f"Scale: {scale}, Shift: {shift}")

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    if file_path.endswith('.npz') :
        output_path = os.path.join(output_dir, file_name)
        output_image_path = os.path.join(output_dir, f"{file_name.split('.')[0]}.png")
    
        depth_data = np.squeeze(np.load(file_path)['pred'])

        print(f"Depth data shape: {depth_data.shape}, min: {depth_data.min()}, max: {depth_data.max()}")

        transformed_depth = apply_scale_and_shift(depth_data, scale, shift)
        print(f"After scaling and shifting, min: {transformed_depth.min()}, max: {transformed_depth.max()}")

        transformed_depth = inverse_depth_transformation(transformed_depth)
        print(f"After inverse transformation, min: {transformed_depth.min()}, max: {transformed_depth.max()}")

        np.savez(output_path, pred=transformed_depth)

        image = np.clip(255.0 / transformed_depth.max() * (transformed_depth - transformed_depth.min()), 0, 255).astype(np.uint8)
        imageio.imwrite(output_image_path, image)

        print(f"Processed and saved: {file_name}")
