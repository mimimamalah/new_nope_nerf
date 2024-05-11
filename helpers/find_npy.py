import numpy as np
import os

# Replace with the actual path to your .npy file
base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
directory = os.path.join(base_path, 'Tanks/Ignatius/poses_bounds.npy')
array = np.load(directory)

print(array)            
