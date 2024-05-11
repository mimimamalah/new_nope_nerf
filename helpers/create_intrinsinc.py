import numpy as np
import os

# Constructing the intrinsic matrix K for the SwissCube dataset
#K = np.array([[607.5730322397038, 0.0, 512.0],
#              [0.0, 607.5730322397038, 512.0],
#              [0.0, 0.0, 1.0]], dtype=np.float32)

# Constructing the intrinsic matrix K for the hubble dataset
K = np.array(   [[1.4067084e+03,   0.0000000e+00,   5.1200000e+02],
                [0.0000000e+00,   1.4067084e+03,   5.1200000e+02],
                [0.0000000e+00,   0.0000000e+00,   1.0000000e+00]], dtype=np.float32)

base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
directory = os.path.join(base_path, "CVLab/hubble_output_resized_paper")
# Saving the matrix to an npz file
np.savez(directory + 'intrinsics.npz', K=K)

print("Intrinsics file saved as intrinsics.npz")
