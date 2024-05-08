import torch
import numpy as np
import plotly.graph_objects as go

def unproject_depth_to_camera_space(depth_map, focal_length, image_size, device):
    height, width = image_size
    i, j = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    # Principal point offset
    cx, cy = width / 2, height / 2
    x0, y0 = width / 2, height / 2  # Principal point offset
    z = depth_map.flatten()  # Depth values
    # Use the provided focal length and principal point offset
    x = (j.flatten() - x0) * z / focal_length
    y = (i.flatten() - y0) * z / focal_length
    points_3d = torch.stack((x, y, z), dim=1).cpu().numpy()
    return points_3d

# Assuming an arbitrary focal length typical for front-facing cameras.
# This value may need to be adjusted for best results.
W, H = 640, 480  # Image dimensions
focal_length = 0.7 * W  # Focal length

# Load your depth map here
depth_map_path = "nope-nerf/out/Tanks/Ignatius/rendering/10000_vis/000741.png"  # Update this path

# Simulate loading a depth map (Replace this with actual loading code)
# Assuming the depth map is stored as a grayscale PNG with depth values
depth_map_np = np.random.rand(H, W)  # Example shape, replace with actual loading code

# Convert the depth map to a torch tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_map_tensor = torch.from_numpy(depth_map_np).to(device)

# Image size (height, width)
image_size = depth_map_tensor.shape

# Unproject depth to 3D points in camera space
point_cloud_camera_space = unproject_depth_to_camera_space(depth_map_tensor, focal_length, image_size, device)

# Assuming point_cloud_camera_space is the array with your 3D points
min_depth, max_depth = np.percentile(point_cloud_camera_space[:, 2], [5, 95])  # Filter out extreme depth values
mask = (point_cloud_camera_space[:, 2] > min_depth) & (point_cloud_camera_space[:, 2] < max_depth)
#filtered_points = point_cloud_camera_space[mask]
filtered_points = point_cloud_camera_space

# Now create the plot
fig = go.Figure(data=[go.Scatter3d(
    x=filtered_points[:, 0],
    y=filtered_points[:, 1],
    z=filtered_points[:, 2],
    mode='markers',
    marker=dict(
        size=2,  # You can change the size of the points here
        opacity=0.8  # Points opacity
    )
)])

# Set axis properties for a more accurate representation
axis_properties = dict(
    showbackground=True,
    backgroundcolor="rgb(230, 230,230)",
    gridcolor="rgb(255, 255, 255)",
    zerolinecolor="rgb(255, 255, 255)"
)

# Set consistent axes range for proper aspect ratio
axes_ranges = dict(
    xaxis=dict(axis_properties, range=[-1, 1]),
    yaxis=dict(axis_properties, range=[-1, 1]),
    zaxis=dict(axis_properties, range=[-1, 1]),
)

fig.update_layout(
    title='3D Point Cloud Visualization',
    scene=dict(
        xaxis_title='X AXIS',
        yaxis_title='Y AXIS',
        zaxis_title='Z AXIS'
    )) #

# Save the figure to HTML
fig.write_html('interactive_point_cloud_2.html')

# If you want to display the plot in a Jupyter notebook as well, uncomment the line below
# fig.show()


import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_mesh_from_point_cloud(point_cloud, voxel_size=0.1):
    # Check if there are enough unique points
    if len(np.unique(point_cloud, axis=0)) < 3:
        raise ValueError("Point cloud does not contain enough unique points.")

    # Define voxel grid
    min_bounds = np.min(point_cloud, axis=0)
    max_bounds = np.max(point_cloud, axis=0)
    grid_shape = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
    grid_origin = min_bounds
    grid_spacing = (max_bounds - min_bounds) / grid_shape

    # Create voxel grid
    grid = np.zeros(grid_shape, dtype=bool)

    # Populate grid with points
    grid_indices = ((point_cloud - grid_origin) / grid_spacing).astype(int)
    grid_indices_clipped = np.clip(grid_indices, 0, grid_shape - 1)
    grid[tuple(grid_indices_clipped.T)] = True

    # Extract mesh using Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(grid, level=0)

    # Translate vertices to world space
    verts = verts * grid_spacing + grid_origin

    return verts, faces


point_cloud = filtered_points

try:
    # Create mesh from point cloud
    verts, faces = create_mesh_from_point_cloud(point_cloud)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='b', marker='o')

    # Plot faces
    for i in range(len(faces)):
        verts_in_face = verts[faces[i], :]
        ax.plot_trisurf(verts_in_face[:, 0], verts_in_face[:, 1], verts_in_face[:, 2], color='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the visualization as a PNG file
    plt.savefig('mesh_visualization.png')

    plt.show()

except ValueError as e:
    print(e)
    print("Unique points in the point cloud:")
    print(np.unique(point_cloud, axis=0))