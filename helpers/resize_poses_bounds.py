import numpy as np
import os

base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
path = os.path.join(base_path, 'KITTI/straight/poses_bounds.npy')
data = np.load(path)

# Assuming the last 3 elements in each row of the pose matrix are the intrinsics
data[:, 12] /= 2  # Update image height
data[:, 13] /= 2  # Update image width
data[:, 14] /= 2  # Update focal length

np.save(path, data)
