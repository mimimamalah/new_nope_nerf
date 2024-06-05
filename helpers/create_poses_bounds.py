import numpy as np

# Define the intrinsic matrix
K = np.array(   [[1.4067084e+03,   0.0000000e+00,   5.1200000e+02],
                [0.0000000e+00,   1.4067084e+03,   5.1200000e+02],
                [0.0000000e+00,   0.0000000e+00,   1.0000000e+00]], dtype=np.float32)

# Generate Euler angles for 360-degree rotation around the z-axis
euler_angle_z = np.linspace(0, 2*np.pi, 500)

# Create camera-to-world transformation matrices
poses = []
for angle in euler_angle_z:
    # Rotation matrix around z-axis
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    # Translation vector adjusted for 26m horizontally and 26m vertically
    T = np.array([26, 0, 26])

    # Camera-to-world matrix (3x4)
    cam_to_world = np.hstack([R, T[:, np.newaxis]])

    # Combine with the intrinsic matrix (reshape K to 3x3 if necessary)
    intrinsic_vector = K.diagonal()[:3]  # Assuming principal points are centered and focal length is same for x and y
    pose_matrix = np.hstack([cam_to_world, intrinsic_vector[:, np.newaxis]])

    # Flatten the matrix and append arbitrary close and far depths
    pose_flat = pose_matrix.flatten()
    pose_with_depths = np.hstack([pose_flat, [0.1, 100]])  # example depth values

    poses.append(pose_with_depths)

# Convert to numpy array and save
poses_bounds = np.array(poses, dtype=np.float32)
np.save('poses_bounds.npy', poses_bounds)

