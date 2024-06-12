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
import tqdm
from plyfile import PlyData, PlyElement

"""
This script extracts corresponding 3D meshes from the NOPE-NeRF model,
utilizing density data to create mesh representations.
Note: Adjust the 'thres', 'bounds_x' (y,z) , and resolution 'res_x' parameters 
according to your specific dataset and accuracy requirements.
"""


def export_point_cloud_ply(density_data, threshold, output_file='point_cloud.ply'):
    # Extract points where the density is greater than the threshold
    z, y, x = np.where(density_data > threshold)
    
    # Create structured data suitable for the PlyElement
    points = np.core.records.fromarrays([x, y, z], dtype=[('x', 'float32'), ('y', 'float32'), ('z', 'float32')])
    ply_element = PlyElement.describe(points, 'vertex')
    
    # Write to PLY file
    PlyData([ply_element], text=True).write(output_file)
    print(f"Point cloud exported to {output_file}")


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

# Generate
model.eval()


render_dir = os.path.join(generation_dir, 'extracted_images')
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

output_mesh = True  # Set this flag to True to output meshes
mesh_dir = os.path.join(render_dir, 'meshes')
os.makedirs(mesh_dir, exist_ok=True)
i = 0
res_x = 200  # Resolution for x-axis
res_y = 200  # Resolution for y-axis
res_z = 200  # Resolution for z-axis
                                           # 3D sampling resolution
#bounds_x = [-0.6, -0.1]  # Example range for x-axis it actually corresponds to the z axis
#bounds_y = [-0.4, 0.6]  # Example range for y-axis
#bounds_z = [-1.3, -0.4]  # Example range for z-axis it is actually the x axis

#res_x = 300  # Resolution for x-axis
#res_y = 300  # Resolution for y-axis
#res_z = 300  # Resolution for z-axis
#For hubble                                           # 3D sampling resolution
#bounds_x = [-0.55, -0.15]  # Example range for x-axis
#bounds_y = [-0.75, -0.45]  # Example range for y-axis
#bounds_z = [-0.45, -0.05]  # Example range for z-axis

bounds_x = [-1.5, 1.5]  # Example range for x-axis
bounds_y = [-1.5, 1.5]  # Example range for y-axis
bounds_z = [-1.5, 1.5]  # Example range for z-axis

black_threshold = torch.tensor([0, 0, 0], device=device)  # RGB black threshold
sphere_radius = 15 
high_coverage_threshold = 1.0
#For Ignatius
thres = 0.6                                              # volume density threshold for marching cubes
#thres = 0.1 
chunk_size = 1024*32  

#with torch.cuda.device(device), torch.no_grad():
#    t_x = torch.linspace(*bounds_x, res_x + 1)
#    t_y = torch.linspace(*bounds_y, res_y + 1)
#    t_z = torch.linspace(*bounds_z, res_z + 1)
#
#    # Generate meshgrid using the new bounds
#    query = torch.stack(torch.meshgrid(t_x, t_y, t_z), dim=-1)
#    query_flat = query.view(-1, 3)

    
#    density_all = []
#    for i in tqdm.trange(0, len(query_flat), chunk_size, leave=False):
#        points = query_flat[None, i:i + chunk_size].to(device)
#        density_samples = model.forward(p=points, only_occupancy it=it)
#        density_all.append(density_samples.cpu())
#    density_all = torch.cat(density_all, dim=1)[0]
#    density_all = density_all.view(*query.shape[:-1]).numpy()



with torch.cuda.device(device), torch.no_grad():
    t_x = torch.linspace(*bounds_x, res_x + 1)
    t_y = torch.linspace(*bounds_y, res_y + 1)
    t_z = torch.linspace(*bounds_z, res_z + 1)

    # Generate meshgrid using the new bounds
    query = torch.stack(torch.meshgrid(t_x, t_y, t_z), dim=-1)
    query_flat = query.view(-1, 3)
    
    density_all = []
    mask_all = []

    for i in tqdm.trange(0, len(query_flat), chunk_size, leave=False):
        points = query_flat[None, i:i + chunk_size].to(device)
        ray_unit = torch.zeros_like(points)
        rgb, density_samples = model.forward(p=points, ray_d=ray_unit, return_addocc=True, it=it)
        # Create binary mask where RGB is considered black
        is_black = (rgb <= black_threshold).all(dim=-1)
        mask_all.append(is_black.cpu())
        #density_samples[is_black] = 0 
        density_all.append(density_samples.cpu())

    # Convert lists to tensors
    #mask_all = torch.cat(mask_all, dim=1)[0].view(*query.shape[:-1])
    density_all = torch.cat(density_all, dim=1)[0].view(*query.shape[:-1]).numpy()

    # Apply 3D Gaussian blur to the mask with a normalization factor
    # Normalizing by the maximum value a fully covered sphere would reach
    #sphere_volume = (4/3) * np.pi * (sphere_radius ** 3)
    #smoothed_mask = scipy.ndimage.gaussian_filter(mask_all.astype(float), sigma=sphere_radius)
    #normalized_mask = smoothed_mask / sphere_volume

    # Threshold the smoothed mask to determine where to zero out density
    #significant_black = normalized_mask > high_coverage_threshold  # Threshold for "significant" black region

    # Zero out density based on the smoothed mask
    #density_all[significant_black] = 0


    # Visualizing and saving multiple slices
    num_slices = 10  # Number of slices you want to visualize
    slice_step = density_all.shape[2] // num_slices  # Step to evenly space the slices

    for slice_index in range(0, density_all.shape[2], slice_step):
        plt.figure()
        plt.imshow(density_all[:,:,slice_index], cmap='gray')
        plt.colorbar()
        plt.title(f"Density Distribution at Slice {slice_index}")
        obj_fname = f"{mesh_dir}/slice_{slice_index}_density.png" 
        plt.savefig(obj_fname)
        plt.close()
    
    #density_all = 1.0 - density_all
    #For Ignatius    
    density_threshold = 0.6
    #density_threshold = 0.1
    obj_fname = f"{mesh_dir}/high_density_points_0.9_2.ply" 
    export_point_cloud_ply(density_all, density_threshold, output_file=obj_fname)


    # Optionally, visualize and save the histogram of density values
    plt.hist(density_all.flatten(), bins=50, log=True)
    plt.title("Density Value Distribution")
    plt.xlabel("Density Values")
    plt.ylabel("Frequency (log scale)")
    obj_fname = f"{mesh_dir}/density_histogram_0.9_2.png" 
    plt.savefig(obj_fname)
    plt.close()

    vertices, triangles = mcubes.marching_cubes(density_all, thres)
    #vertices_centered = vertices / res - 0.5
    mesh = trimesh.Trimesh(vertices, triangles)
    
    obj_fname = f"{mesh_dir}/mesh_0.9_2.ply"
    mesh.export(obj_fname)


