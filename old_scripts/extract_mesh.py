import torch
import numpy as np
import plotly.graph_objects as go

def unproject_depth_to_camera_space(depth_map, camera_intrinsics, device):
    height, width = depth_map.shape
    i, j = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    z = depth_map.flatten()  # Depth values
    x = (j.flatten() - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]  # X coordinates in camera space
    y = (i.flatten() - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]  # Y coordinates in camera space
    points_3d = torch.stack((x, y, z), dim=1).cpu().numpy()
    return points_3d

# Paths to your depth map and camera intrinsics
depth_map_path = "nope-nerf/out/Tanks/Ignatius/rendering/10000_vis/000741.png"
camera_intrinsics_path = "data/CVLab/SwissCube_cropped/intrinsics.npz"

# Load depth map and camera intrinsics
depth_data = np.load(depth_map_path)
camera_intrinsics_data = np.load(camera_intrinsics_path)

depth_map_np = depth_data['pred']  # Assuming the depth map is stored under the key 'pred'
camera_intrinsics_np = camera_intrinsics_data['K']  # Assuming intrinsics are stored under the key 'K'

# Convert numpy arrays to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_map_tensor = torch.from_numpy(depth_map_np).to(device)
camera_intrinsics_tensor = torch.from_numpy(camera_intrinsics_np).float().to(device)

# Unproject depth to 3D points in camera space
point_cloud_camera_space = unproject_depth_to_camera_space(depth_map_tensor, camera_intrinsics_tensor, device)

# Assuming point_cloud_camera_space is the array with your 3D points
min_depth, max_depth = np.percentile(point_cloud_camera_space[:, 2], [5, 95])  # Filter out extreme depth values
mask = (point_cloud_camera_space[:, 2] > min_depth) & (point_cloud_camera_space[:, 2] < max_depth)
filtered_points = point_cloud_camera_space[mask]

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
        zaxis_title='Z AXIS',
        **axes_ranges,
        aspectmode='cube'  # This ensures equal aspect ratio
    ),
    margin=dict(r=0, b=0, l=0, t=0)) #

# Save the figure to HTML
fig.write_html('interactive_point_cloud.html')

# If you want to display the plot in a Jupyter notebook as well, uncomment the line below
# fig.show()
