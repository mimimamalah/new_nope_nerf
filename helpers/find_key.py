import numpy as np
import os

def inspect_npz_file(npz_file_path):
    # Load the NPZ file
    with np.load(npz_file_path) as npz_file:
        print(f"Inspecting '{npz_file_path}':")

        # Iterate over the items in the NPZ file
        for key, value in npz_file.items():
            print(f"  Key: {key}, Shape: {value.shape}")
            
            print("Max:", np.max(value))
            print("Min:", np.min(value))
            
            # Decide if to print the contents based on array size
            if value.size <= 10:  # Adjust this threshold as needed
                print("  Contents:", value)
            else:
                # For larger arrays, print a sample
                print("  Sample contents (first 5 elements):", value.flat[100400:100420])

        # Optionally access specific keys directly (demonstration purpose)
        # Example: Uncomment to check for a specific key, such as 'pred'
        if 'pred' in npz_file:
            print("\n  Specific key 'pred': Shape:", npz_file['pred'].shape)
            print("  Contents:", npz_file['pred'] if npz_file['pred'].size <= 100 else "Large array, contents not fully printed")


base_path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Paths to NPZ files you want to inspect
npz_file_paths = [
    os.path.join(base_path, "CVLab/hubble_output_resized/intrinsics.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_paper/intrinsics.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_correct/intrinsics.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_correct_anything/intrinsics.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_paper/dpt/depth_0000.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_correct/dpt/depth_0000.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized/dpt/depth_0000.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_correct/dpt-before/depth_0000.npz"), 
    os.path.join(base_path, "CVLab/hubble_output_resized_paper/dpt/depth_0045.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized_correct/dpt/depth_0045.npz"),
    os.path.join(base_path, "CVLab/hubble_output_resized/dpt/depth_0045.npz"), 
    os.path.join(base_path, "CVLab/hubble_output_resized_correct/dpt-before/depth_0045.npz"),
    os.path.join(base_path, "KITTI/straight-anything-scale-original/dpt-anything/depth_00000.npz"), 
    os.path.join(base_path, "KITTI/straight-anything-scale-original/dpt-not-inversed/depth_00000.npz"), 
    os.path.join(base_path, "Tanks/Ignatius/dpt/depth_000741.npz"),
    #os.path.join(base_path, "CVLab/hubble_output_resized_correct_anything/dpt-inversed/depth_0000.npz"),
    #os.path.join(base_path, "CVLab/hubble_output_resized_correct_anything/dpt-not-inversed/depth_0000.npz"),

    #os.path.join(base_path, "CVLab/hubble_output_complete/dpt/depth_0145.npz"),
    #os.path.join(base_path, "CVLab/hubble_output_resized_paper/dpt/depth_0145.npz"),
]

# Inspect each NPZ file in the list
for path in npz_file_paths:
    print(path)
    inspect_npz_file(path)
    print("-" * 60)  # Separator for readability
