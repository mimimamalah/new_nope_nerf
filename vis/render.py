import os
import sys
import argparse
import time
import torch
import cv2
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.common import convert3x4_4x4,  interp_poses, interp_poses_bspline, generate_spiral_nerf
from model.extracting_images import Extract_Images
import model as mdl
import imageio
import numpy as np
import trimesh
import mcubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
from scipy.ndimage import uniform_filter
import torch.nn.functional as F
import vtk
import scipy.ndimage

def save_point_cloud(points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assume that points[:, 0], points[:, 1], and points[:, 2] are the RGB values
    # and points[:, 3], points[:, 4], points[:, 5] are the XYZ coordinates.
    # Normalize RGB values for Matplotlib (which expects [0, 1] range)
    colors = points[:, :3] / 255

    ax.scatter(points[:, 3], points[:, 4], points[:, 5], c=colors, marker='o')
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Z Coordinates')
    plt.savefig(filename)  # Save the plot to a file
    plt.close(fig)

def calculate_box_dimensions(point_cloud):

    min_coord = np.min(point_cloud, axis=0)
    max_coord = np.max(point_cloud, axis=0)
    dimensions = max_coord - min_coord
    box_length = np.max(dimensions)

    volume = len(point_cloud) * (box_length / len(point_cloud))**3

    return box_length, volume

def depth_to_voxel_grid(depth_image, depth_min, depth_max, grid_size):
    """
    Converts a 2D depth image into a 3D voxel grid.

    Args:
        depth_image (numpy array): The 2D array of depth values.
        depth_min (float): The minimum depth to be considered valid.
        depth_max (float): The maximum depth to be considered valid.
        grid_size (int): The resolution of the grid in the depth dimension.

    Returns:
        numpy.ndarray: A 3D numpy array representing the voxel grid.
    """
    # Initialize the voxel grid
    height, width = depth_image.shape
    voxel_grid = np.zeros((height, width, grid_size))

    # Scale the depth values to the grid size
    scaled_depths = np.clip((depth_image - depth_min) / (depth_max - depth_min) * grid_size, 0, grid_size - 1).astype(int)

    # Activate the voxels according to their scaled depth
    for i in range(height):
        for j in range(width):
            if depth_min <= depth_image[i, j] <= depth_max:
                voxel_grid[i, j, scaled_depths[i, j]] = 1

    return voxel_grid


def depth_to_mesh(volume):
    """
    Converts a 3D volume into a mesh using the marching cubes algorithm.

    Args:
        volume (numpy array): The 3D array (volume) of the object.

    Returns:
        trimesh.Trimesh: The generated mesh.
    """
    # Use marching cubes to create the mesh
    verts, faces = mcubes.marching_cubes(volume, 0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh


def generate_point_cloud(rgb_image, depth_image, camera_matrix, depth_scale):
    """
    Generate a point cloud from a depth image and RGB image using the camera matrix.

    Args:
        rgb_image (numpy array): The RGB image.
        depth_image (numpy array): The depth image.
        camera_matrix (numpy array): The camera matrix, assumed to be a 4x4 transformation matrix.
        depth_scale (float): Scale factor to convert depth image values to meters.

    Returns:
        numpy.ndarray: An array of shape (N, 6) where N is the number of points and each point has RGBXYZ.
    """
    # Adjust the camera matrix if it has an extra dimension and move it to CPU
    if camera_matrix.dim() > 2:
        camera_matrix = camera_matrix.squeeze(0)  # Remove extraneous dimensions if any
    if camera_matrix.is_cuda:
        camera_matrix = camera_matrix.cpu().numpy()  # Convert to numpy after ensuring it's on CPU
    else:
        camera_matrix = camera_matrix.numpy()

    # Extract pseudo intrinsic parameters from the camera matrix
    fx = camera_matrix[0, 0]
    fy = abs(camera_matrix[1, 1])  # Use absolute because of negative scaling
    cx = camera_matrix.shape[1] // 2  # Assuming center of the image if not provided
    cy = camera_matrix.shape[0] // 2

    # Ensure the depth image is in float and scale it to meters
    depth_image = depth_image.astype(np.float32) * depth_scale

    # Get the dimensions of the depth image
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Backproject to 3D space
    x = (u - cx) * depth_image / fx
    y = (v - cy) * depth_image / fy
    z = depth_image

    # Stack the coordinates together
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Filter out invalid points (where depth is zero)
    valid_mask = depth_image.flatten() > 0
    points = points[valid_mask]

    # Convert RGB image to RGB format if it's not already
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    colors = rgb_image.reshape(-1, 3)[valid_mask]  # Reshape and mask color to valid points

    # Concatenate points and colors
    point_cloud = np.hstack((colors, points))  # shape (N, 6) with RGBXYZ

    return point_cloud

def invert_and_extract_mesh(volume, isovalue, pivot):
    
    #subvolume = volume[subvolume_bounds[0][0]:subvolume_bounds[0][1], 
    #                   subvolume_bounds[1][0]:subvolume_bounds[1][1], 
    #                   subvolume_bounds[2][0]:subvolume_bounds[2][1]]

    # Invert the subvolume data based on the pivot
    #inverted_subvolume = pivot + (pivot - subvolume)
    
    #lower_threshold = 0.1
    #upper_threshold = 0.6

    # Create a mask where only the values within the threshold range are True
    #mask = (volume > lower_threshold) & (volume < upper_threshold)

    # Apply the mask to the volume
    #processed_volume = np.zeros_like(volume)
    #processed_volume[mask] = volume[mask]

    # Find the range of the inverted subvolume
    #min_value = np.amin(subvolume)
    #max_value = np.amax(subvolume)
    #print("Range of inverted subvolume data: Min =", min_value, "Max =", max_value)

    # Use marching cubes to create the mesh from the inverted subvolume
    verts, faces = mcubes.marching_cubes(volume, isovalue)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    # Reverse the order of vertices in each face to flip the normals
    #mesh.faces = mesh.faces[:, ::-1]

    # Explicitly reverse the normals
    #mesh.vertex_normals = -mesh.vertex_normals

    return mesh


def make_mesh_hollow(volume, scale_factor):
    verts, faces = mcubes.marching_cubes(volume, 0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Calculate the centroid
    centroid = mesh.centroid
    
    # Create a hollow version by scaling the mesh inward or outward
    hollow_verts = (mesh.vertices - centroid) * scale_factor + centroid
    
    # Create a new mesh with the hollow vertices but same faces
    # Optionally, reverse the faces if scaled inward to keep normals outward
    if scale_factor < 1:
        hollow_faces = faces[:, ::-1]
    else:
        hollow_faces = faces
    
    hollow_mesh = trimesh.Trimesh(vertices=hollow_verts, faces=hollow_faces)

    # Combine original and hollow mesh to create a shell
    combined_mesh = trimesh.util.concatenate(mesh, hollow_mesh)

    return combined_mesh


def filter_alpha_by_colorfulness(alpha, rgb, radius=15, color_threshold=0.01, darkness_threshold=0.05):
    # Convert RGB to grayscale
    grayscale = rgb2gray(rgb)

    # Compute a mask of pixels that are darker than the darkness_threshold
    dark_pixels_mask = grayscale < darkness_threshold

    # Use a uniform filter to compute the local average of dark pixels over a circular window
    # The uniform filter acts as an averaging filter where we consider the radius around each pixel
    local_darkness = uniform_filter(dark_pixels_mask.astype(float), size=radius*2, mode='constant', cval=0)

    # Compute a mask where the local darkness exceeds the color_threshold (e.g., 99% of the area is dark)
    low_color_area_mask = local_darkness > (1 - color_threshold)

    # Set alpha values to 0 where the mask is True, across all layers
    for layer in range(alpha.shape[2]):
        alpha[:, :, layer] = np.where(low_color_area_mask, 0, alpha[:, :, layer])

    return alpha

def apply_transformation_and_reconstruct(alpha_pred, transformation_mat, device):
    # Ensure alpha_pred is a PyTorch tensor
    if isinstance(alpha_pred, np.ndarray):
        alpha_pred = torch.from_numpy(alpha_pred).float().to(device)

    # Define the grid size
    h, w, d = alpha_pred.shape

    # Generate the grid for sampling, expects normalized coordinates [-1, 1]
    grid = F.affine_grid(transformation_mat[:3].unsqueeze(0), [1, 1, h, w, d], align_corners=True)

    # Interpolate using grid_sample
    new_volume = F.grid_sample(alpha_pred.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return new_volume.squeeze(0)

def back_project_to_volume(volume, density_map, view_matrix, slice_thickness=1):
    """
    Project a 2D density map into a 3D volume based on the view matrix.
    """
    # Simplified example where z_start is determined from view_matrix
    z_index = int(view_matrix[2, 3] * 100)  # Example transformation, scale appropriately

    # Ensure z_index is within the bounds of the volume depth
    if 0 <= z_index < volume.shape[2]:
        volume[:, :, z_index] += density_map  # Add density map to the slice at z_index

def normalize_volume(volume):
    """
    Normalize the volume data after all back-projections.
    """
    max_density = np.max(volume)
    if max_density > 0:
        volume /= max_density 


def resample_density(density, world_mat, camera_mat, scale_mat, volume_size):
    camera_mat = torch.inverse(camera_mat).cpu().numpy()
    world_mat = torch.inverse(world_mat).cpu().numpy()
    scale_mat = torch.inverse(scale_mat).cpu().numpy()
    
    print("World Matrix: ", world_mat)
    
    # Compute the composite transformation matrix
    view_matrix = np.dot(scale_mat, np.dot(world_mat, camera_mat))
    print("View Matrix: ", view_matrix)

    # Extract rotation and scaling, translation components
    transform_matrix = view_matrix[:3, :3]
    offset = view_matrix[:3, 3]

    # Scale the translation component to better fit the volume size
    scale_factors = [volume_size[i] / density.shape[i] for i in range(3)]
    scaled_offset = offset * scale_factors

    print("Scaled Offset: ", scaled_offset)

    # Convert density to numpy if it's a tensor
    density = density.cpu().numpy() if isinstance(density, torch.Tensor) else density

    # Apply affine transformation with scaled translation
    resampled_density = scipy.ndimage.affine_transform(
        density,
        matrix=transform_matrix,
        offset=scaled_offset,  # use the scaled offset here
        output_shape=volume_size,
        order=1,
        mode='constant',
        cval=0
    )

    return resampled_density


def save_volume_to_vtk(volume, filename):
    # Create a VTK image data object
    image_data = vtk.vtkImageData()

    # Set the volume dimensions
    image_data.SetDimensions(volume.shape)

    # Prepare data type
    if volume.dtype == np.float32:
        vtk_type = vtk.VTK_FLOAT
    elif volume.dtype == np.float64:
        vtk_type = vtk.VTK_DOUBLE
    elif volume.dtype == np.int:
        vtk_type = vtk.VTK_INT
    else:
        raise ValueError("Unsupported data type")

    # Set the type of data
    image_data.AllocateScalars(vtk_type, 1)

    # Get the pointer to the volume data
    pts = image_data.GetPointData().GetScalars()

    # Copy the data into the VTK image data structure
    for i in range(volume.size):
        pts.SetTuple1(i, volume.flatten()[i])

    # Create a writer for the data
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)

    # Check if we need to write binary or ASCII file
    writer.SetFileTypeToBinary()
    writer.SetInputData(image_data)
    writer.Write()



torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Extract images.'
)
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['extract_images']['extraction_dir'])

# Model
model_cfg = cfg['model']
network_type = cfg['model']['network_type']
if network_type=='official':
    model = mdl.OfficialStaticNerf(cfg)

rendering_cfg = cfg['rendering']
renderer = mdl.Renderer(model, rendering_cfg, device=device)

# init model
nope_nerf = mdl.get_model(renderer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=nope_nerf)
load_dict = checkpoint_io.load(cfg['extract_images']['model_file'])
it = load_dict.get('it', -1)

op = cfg['extract_images']['traj_option']
N_novel_imgs = cfg['extract_images']['N_novel_imgs']

train_loader, train_dataset = get_dataloader(cfg, mode='render', shuffle=False, n_views=N_novel_imgs)
n_views = train_dataset['img'].N_imgs

if cfg['pose']['learn_pose']:
    if cfg['pose']['init_pose']:
        init_pose = train_dataset['img'].c2ws 
    else:
        init_pose = None
    pose_param_net = mdl.LearnPose(n_views, cfg['pose']['learn_R'], cfg['pose']['learn_t'], cfg=cfg, init_c2w=init_pose).to(device=device)
    checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=pose_param_net)
    checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'])
    learned_poses = torch.stack([pose_param_net(i) for i in range(n_views)])
    
    if op=='sprial':
        bds = np.array([2., 4.])
        hwf = train_dataset['img'].hwf
        c2ws = generate_spiral_nerf(learned_poses, bds, N_novel_imgs, hwf)
        c2ws = convert3x4_4x4(c2ws)
    elif op =='interp':
        c2ws = interp_poses(learned_poses.detach().cpu(), N_novel_imgs)
    elif op=='bspline':
        i_train = train_dataset['img'].i_train
        degree=cfg['extract_images']['bspline_degree']
        c2ws = interp_poses_bspline(learned_poses.detach().cpu(), N_novel_imgs, i_train,degree)

c2ws = c2ws.to(device)
if cfg['pose']['learn_focal']:
    focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
    checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
    checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'])
    fxfy = focal_net(0)
    print('learned fx: {0:.2f}, fy: {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
else:
    fxfy = None
# Generator
generator = Extract_Images(
    renderer,cfg,use_learnt_poses=cfg['pose']['learn_pose'],
    use_learnt_focal=cfg['pose']['learn_focal'],
    device=device, render_type=cfg['rendering']['type']
)

# Generate
model.eval()

render_dir = os.path.join(generation_dir, 'extracted_images', op)
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

imgs = []
depths = []
geos = []
alpha = []
camera_mats = []
world_mats = []
scale_mats = []
densities = []
output_geo = True

output_mesh = False  # Set this flag to True to output meshes
mesh_dir = os.path.join(render_dir, 'meshes')
os.makedirs(mesh_dir, exist_ok=True)
i = 0

volume_size = (768, 768, 768) 
volume = np.zeros(volume_size, dtype=np.float32)

for data in train_loader:
    #if i < 10 :
        #print("Data : ", data)
        #print("Data rendered_dir : ", render_dir)
        #print("Data c2ws : ", c2ws)
        #print("Data fxfy : ", fxfy)
        #print("Data it : ", it)
        #print("Data output_geo : ", output_geo)
        out, img_idx = generator.generate_images(data, render_dir, c2ws, fxfy, it, output_geo)
        imgs.append(out['img'])
        depths.append(out['depth'])
        geos.append(out['geo'])
        alpha.append(out['alpha'])
        camera_mats.append(out['camera_mat'])
        world_mats.append(out['world_mat'])
        scale_mats.append(out['scale_mat'])
        #print("world_mats shape" , out['world_mat'].shape)
        #print("world_mats values" , out['world_mat'])
        #print("camera_mats shape" , out['camera_mat'].shape)
        #print("camera_mats values" , out['camera_mat'])
        #print("scale_mats shape" , out['scale_mat'].shape)
        #print("scale_mats values" , out['scale_mat'])
        if output_mesh:
            depth_image = out['depth']
            #print("Depth values: ", out['depth'])
            #np.savetxt('depth_values.txt', out['depth'], fmt='%f')
            #print("Imgs values : ", out['img'])
            #print("Geo values : ", out['geo'])
            #print("Alpha values : ", out['alpha'])
            #print("Camera values : ", out['camera'])
            #np.savetxt('alpha_values.txt', out['alpha'], fmt='%f') 
            #point_cloud = generate_point_cloud(out['img'], out['depth'], out['camera'], 1000)
            #save_point_cloud(point_cloud, 'point_cloud_plot.png')
            #box_length, volume = calculate_box_dimensions(point_cloud)
            #print('Box Length:', box_length)
            #print('Box Volume:', volume)
            #volume = depth_to_voxel_grid(depth_image, depth_min=1, depth_max=255, grid_size=100)
            alpha_i = out['alpha']
            alpha_i = filter_alpha_by_colorfulness(alpha_i, out['img'])
            subvolume_bounds = ([120, 650], [120, 650], [0, 128])
            full_volume = np.zeros_like(alpha_i)
            full_volume[subvolume_bounds[0][0]:subvolume_bounds[0][1], 
                        subvolume_bounds[1][0]:subvolume_bounds[1][1], 
                        subvolume_bounds[2][0]:subvolume_bounds[2][1]] = alpha_i[subvolume_bounds[0][0]:subvolume_bounds[0][1], 
                                                                                subvolume_bounds[1][0]:subvolume_bounds[1][1], 
                                                                                subvolume_bounds[2][0]:subvolume_bounds[2][1]]

            # Now, full_volume contains the subvolume data with the rest of the space filled with zeros
            #alpha_i = np.sum(alpha_i, axis=2)
            densities.append(full_volume)
            #print('some values of alpha', alpha_i[:20])
            #print('out alpha shape : ', alpha_i.shape)
            #mesh = depth_to_mesh(alpha_i)
            # Define the pivot value for inversion
            #pivot_value = 0.005
            # Define the isovalue for the marching cubes
            #isovalue = 0.005
            # Assuming 'volume' is your volume data and is properly set up
            # This function call will return the mesh for the specified subvolume with inverted data
            #inverted_mesh = invert_and_extract_mesh(alpha_i, isovalue, pivot_value)
            #hollow_mesh = make_mesh_hollow(out['alpha'], 0.9)
            
             
            #density_slice = alpha_i[160:600, 160:600, 120]

            # Plot the density values
            #plt.figure(figsize=(10, 8))
            #plt.imshow(density_slice, cmap='viridis')  # Using 'viridis' colormap, change as needed
            #plt.colorbar(label='Density Value')
            #plt.title('Density Slice at Index 120')
            #output_file_path = f"{img_idx}_density_slice_120.png"
            #plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
            #plt.close()
            
            #density_slice = alpha_i[:,:,40]

            # Plot the density values
            #plt.figure(figsize=(10, 8))
            #plt.imshow(density_slice, cmap='viridis')  # Using 'viridis' colormap, change as needed
            #plt.colorbar(label='Density Value')
            #plt.title('Density 40')
            #output_file_path = f"{img_idx}_density_40.png"
            #plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
            #plt.close()
            
            #density_slice = alpha_i[160:600, 160:600, 99]

            # Plot the density values
            #plt.figure(figsize=(10, 8))
            #plt.imshow(density_slice, cmap='viridis')  # Using 'viridis' colormap, change as needed
            #plt.colorbar(label='Density Value')
            #plt.title('Density Slice at Index 99')
            #output_file_path = f"{img_idx}_density_slice_99.png"
            #plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
            #plt.close()
             
            #mesh_file = os.path.join(mesh_dir, f"mesh_{img_idx}.ply")
            #mesh.export(mesh_file)
            #mesh_file_inverted = os.path.join(mesh_dir, f"mesh_{img_idx}_inverted.ply")
            #inverted_mesh.export(mesh_file_inverted)
            #hollow_mesh_file = os.path.join(mesh_dir, f"mesh_{img_idx}_hollow.ply")
            #hollow_mesh.export(hollow_mesh_file)
                
                     
   #i+=1

#world_mats_tensor = torch.stack([mat.squeeze(0) for mat in world_mats])
#camera_mats_tensor = torch.stack([mat.squeeze(0) for mat in camera_mats])
#scale_mats_tensor = torch.stack([mat.squeeze(0) for mat in scale_mats])
#inverse_world_mats = torch.inverse(world_mats_tensor)

# Applying transformations to each alpha_pred
#transformed_densities = [apply_transformation_and_reconstruct(alpha, inv_mat,device) for alpha, inv_mat in zip(densities, inverse_world_mats)]

#stacked_densities = torch.stack(transformed_densities, dim=0)

#average_density = torch.mean(stacked_densities, dim=0)

#for density_map, world_mat, camera_mat, scale_mat in zip(densities, world_mats, camera_mats, scale_mats):
    # Assume densities and volume are numpy arrays and view_matrix is a 4x4 numpy array
#    transformed_density = resample_density(density_map, world_mat, camera_mat, scale_mat, volume_size)
#    volume += transformed_density  
    

#normalize_volume(volume)
#pivot_value = 0.002
# Define the isovalue for the marching cubes
#isovalue = 0.002
# Assuming 'volume' is your volume data and is properly set up
# This function call will return the mesh for the specified subvolume with inverted data
#full_density = np.concatenate(densities, axis=2)
#inverted_mesh = invert_and_extract_mesh(volume, isovalue, pivot_value)   
#mesh_file_inverted = os.path.join(mesh_dir, f"mesh_total_inverted.ply")
#inverted_mesh.export(mesh_file_inverted)
    
#print("Same alphas or not", alpha[0] == alpha[1])
    
        
imgs = np.stack(imgs, axis=0)
depths = np.stack(depths, axis=0)

video_out_dir = os.path.join(render_dir, 'video_out')
if not os.path.exists(video_out_dir):
    os.makedirs(video_out_dir)
imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)
imageio.mimwrite(os.path.join(video_out_dir, 'depth.mp4'), depths, fps=30, quality=9)
if output_geo:  
    geos = np.stack(geos, axis=0)
    imageio.mimwrite(os.path.join(video_out_dir, 'geo.mp4'), geos, fps=30, quality=9)


       
