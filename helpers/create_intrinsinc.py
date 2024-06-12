import numpy as np
import os

"""
This script generates the correct intrinsic camera matrix file.
If you changed the resolution of the images, you need to adjust the intrinsic matrix accordingly.
You can uncoment the lines for adjusting according to the previous and new resolution.
"""

# Constructing the intrinsic matrix K for the SwissCube dataset
#K = np.array([[607.5730322397038, 0.0, 512.0],
#              [0.0, 607.5730322397038, 512.0],
#              [0.0, 0.0, 1.0]], dtype=np.float32)

# Constructing the intrinsic matrix K for the hubble dataset
K = np.array(   [[1.4067084e+03,   0.0000000e+00,   5.1200000e+02],
                [0.0000000e+00,   1.4067084e+03,   5.1200000e+02],
                [0.0000000e+00,   0.0000000e+00,   1.0000000e+00]], dtype=np.float32)

# Original and new resolutions
#old_resolution = (1024, 1024)
#new_resolution = (768, 768)

# Calculate scale factor
#scale_factor = new_resolution[0] / old_resolution[0]  # Both dimensions have the same scale

# Adjust the focal lengths and the principal points
#K[0, 0] *= scale_factor  # fx
#K[1, 1] *= scale_factor  # fy
#K[0, 2] *= scale_factor  # cx
#K[1, 2] *= scale_factor  # cy

# Base path setup
base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
#directory = os.path.join(base_path, "CVLab/hubble_output_resized_paper")
directory = os.path.join(base_path, "CVLab/hubble_output_resized_correct")

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Save the adjusted matrix to an npz file
np.savez(os.path.join(directory, 'intrinsics.npz'), K=K)

print("Intrinsics file saved as intrinsics.npz")
