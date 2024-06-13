# Unkown Object Extraction from Multi-view Synthetic Images with No Pose Prior

**[Data](https://drive.google.com/drive/folders/19Ya9OokrFaWM6nhdBPZzUFo1PT38p1Jg?usp=share_link) | [PDF]() | [Pretrained Models]()** | [Slides](https://docs.google.com/presentation/d/1_qrccUqPkM38neG3Gj4-8cIXEXCuI2L5/edit#slide=id.g2e46917510a_0_83)

## Table of Content
- [Unkown Object Extraction from Multi-view Synthetic Images with No Pose Prior](#unkown-object-extraction-from-multi-view-synthetic-images-with-no-pose-prior)
	- [Table of Content](#table-of-content)
	- [Overview](#overview)
	- [Installation](#installation)
	- [Data](#data)
	- [Monocular Depth Estimation](#monocular-depth-estimation)
	- [Training](#training)
	- [Visualisations](#visualisations)
	- [Evaluation](#evaluation)
	- [Issues](#issues)
	- [Acknowledgement](#acknowledgement)


## Overview
This is the codebase for my semester project.
We aim extract mesh surface from satellite images by addressing challenges encountered with the Neural Radiance Fields (NeRF) approach, particularly under the NoPe-NeRF configuration, which solely uses images without additional parameters. Tests using different datasets revealed that NoPe-NeRF underperforms with non- consecutive images, low-light, and low-texture conditions, as well as with synthetic data. To overcome these limitations, we used a novel monocular depth estimation method that replaces the traditional Depth Prediction Transformer (DPT) with our Depth-Anything model. This adaptation significantly boosts performance in challenging scenarios, evidenced by improved pose metrics and Peak Signal-to- Noise Ratio (PSNR), though it shows minimal impact on high- texture real-world datasets. Additionally, we explored the reconstruction of mesh surfaces. Our findings demonstrate that the combined use of NoPe-NeRF and Depth-Anything models substantially enhances the accuracy and feasibility of 3D reconstructions.
My implementation is built on top of [NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior](https://github.com/ActiveVisionLab/nope-nerf/).

## Installation

Using Scitas

Need to connect first using EPFL VPN :
```
ssh −X username@izar.epfl.ch
```

```
git clone https://github.com/mimimamalah/new_nope_nerf.git
cd new_nope_nerf
python −m venv nope_nerf_env source nope_nerf_env
source nope_nerf_env/bin/activate
source env.sh
pip install -r requirement.txt
```

To synchronize the directory dir from SCITAS to your computer: 
```
rsync -azP username@izar.epfl.ch:dir/ dir
```

From your computer to SCITAS :
```
rsync -azP dir/ username@izar.epfl.ch:dir
```

There are compatibility issues with SCITAS, so it's advisable to contact them to ensure there are no conflicts. You should only load modules that are already installed. You can find modules using:
```
module spider elem
module load elem
```

## Data
1. [Ignatius Dataset](https://drive.google.com/drive/folders/15GMJsi0Bo4jcMkBPGJG2PoJVuKUUzxVA?usp=share_link):
   We used the Ignatius dataset for a real-world, high texture, good lighting and consecutive cataset.

2. [Stove Dataset](https://drive.google.com/drive/folders/19z2JRqa7lxev920sADRDi4YjGxDQF8wG?usp=share_link):
   We used the stove dataset for a real-world, high texture, bad lighting and consecutive cataset.
   
3. [Hot-Dog Dataset](https://drive.google.com/drive/folders/1UsoMP42vHM_6IA3O5N0Uphk87uuTR3ih?usp=share_link):
   We used the Hot-dog dataset for a synthetic, low texture, good lighting and non consecutive dataset.
   
4. [V-KITTI Dataset](https://drive.google.com/drive/folders/1psox4loqxtOYbdm93ZwiTHA5wepOfqdV?usp=share_link):
   We used the straight dataset for a synthetic, high texture, good lighting and consecutive dataset.
   
5. [Hubble Dataset](https://drive.google.com/drive/folders/1vFdLv0rQ9-Y96cveDZ2FbK3T48NhKZis?usp=share_link):
   We used the hubble dataset for a synthetic, low texture, good lighting and consecutive dataset.

6. [Dark Hubble Dataset](https://drive.google.com/drive/folders/1vPt4Mzi3eOkJFZFphC8HBLnX7DahksUE?usp=share_link):
   We used the dark hubble dataset for a synthetic, low texture, bad lighting and consecutive dataset. 

7. If you want to use your own image sequence with customised camera intrinsics, you need to add an `data/your_folder/your_scene/intrinsics.npz` file to the scene directory. 
You can run the helpers/create_intrinsic.py file.
   
One example of the config file is `configs/your_folder/your_scene/images.yaml` (please add your own data to the `data/your_folder/your_scene/images` directory). 


## Monocular Depth Estimation
If you want to use Depth-Anything :
You need to clone this repo first https://github.com/mimimamalah/new_depth_anything.git :
Don't forget to customise this command with your own folder name and scene name.
```

cd ~/new_depth_anything/

python run.py --encoder vitl --img-path ../new_nope_nerf/data/your_folder/your_scene\
//images --outdir ../new_nope_nerf/data/your_folder/your_scene\
//dpt-anything --pred-only

```

Then you need to come back to your new_nope_nerf directory :
```
python depth_anything_scripts/adapt_anything.py configs/your_folder/your_scene/images.yaml

```


If you have a very specific type of data you may need to find your own distortion parameters : scale and shift.
By first running DPT where you should add these two parameters in the pre process config file : 
```
folder_name: 'dpt-before'
invert: False
```

then run 
```
python preprocess/dpt_depth.py configs/your_folder/your_scene/preprocess-for-anything.yaml
```

In this file you need to change where you save your scale and shift factor directory :
```
depth_anything_dir = "data/your_folder/your_scene/dpt-anything"
depth_original_dir = "data/your_folder/your_scene/dpt-before"
transformed_dir = "depth_anything_scripts/your_new_scale_and_shift"
```

then run 
```
python initial_scripts_anything/find_scale_and_shift.py
```

then you can come back to run the adapt_anything.py file where you should change your new scale and shift folder.


If you want to use the original DPT 
Monocular depth map generation: you can first download the pre-trained DPT model from [this link](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing) provided by [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) to `DPT` directory, then run
```
python preprocess/dpt_depth.py configs/your_folder/your_scene/preprocess.yaml
```
to generate monocular depth maps. You need to modify the `cfg['dataloading']['path']` and `cfg['dataloading']['scene']` in `configs/your_folder/your_scene/preprocess.yaml` to your own image sequence.

## Training

1. Train a new model from scratch:

```
python train.py configs/your_folder/your_scene/images.yaml`
```
where you can replace `configs/your_folder/your_scene/images.yaml` with other config files.

You can monitor on <http://localhost:6006> the training process on your own computer, better than using SCITAS using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ./out --port 6006
```
An example of my corresponding batch script on SCITAS :

```
#!/bin/bash                                                                     
#SBATCH --chdir /home/username                                                  
#SBATCH --partition=gpu                                                         
#SBATCH --qos=gpu                                                               
#SBATCH --gres=gpu:1                                                            
#SBATCH --nodes=1                                                               
#SBATCH --time=0-15:00:00                                                       
#SBATCH --ntasks-per-node=1                                                     
#SBATCH --cpus-per-task=8                                                       
#SBATCH --mem 32G                                                               

echo STARTED  at `date`

cd ~/new_nope_nerf/

python train.py configs/your_folder/your_scene/images.yaml

echo FINISHED at date
```

## Visualisations
Mesh
You need to adjust your own iso_value, bounds and resolution.
```
python vis/render_mesh.py configs/your_folder/your_scene/images.yaml
```

Novel view synthesis
```
python vis/render.py configs/your_folder/your_scene/images.yaml
```
Pose visualisation (estimated trajectory only)
```
python vis/vis_poses.py configs/your_folder/your_scene/images.yaml
```


For available training options, please take a look at `configs/default.yaml`.
## Evaluation
For evaluation you need to add your own `data/your_folder/your_scene/poses_bounds.npy` file.
The poses_bounds.npy file contains a numpy array of size N × 17, where N is the number of input images.
 Each row consists of:
• A 3x4 camera-to-world affine transformation matrix.
• A 1x3 vector containing the image height, width, and focal length.
• Two depth values representing the closest and farthest scene content from that point of view.
Each row of length 17 gets reshaped into a 3x5 pose matrix and two depth values that bound the scene content.

1. Evaluate image quality and depth:
```
python evaluation/eval.py configs/your_folder/your_scene/images.yaml
```
To evaluate depth: add `--depth` . Note that you need to add ground truth depth maps by yourself.


2. Evaluate poses:
To evaluate poses on SCITAS, you need to uninstall open3d and then reinstall it when you need it.
```
python evaluation/eval_poses.py configs/your_folder/your_scene/images.yaml
```

You can only viusalize it on your own computer not scitas
To visualise estimated & ground truth trajectories: add `--vis` 


## Issues
If you have an OOM error, you either neeed to increase your mem from 32G to a bigger value. It it is still not enough, you may need to reduce your image resolution. For example for the hubble we had to reduce resolution from 1024*1024 to 768*768. For the stove dataset, we used 80G. Requesting more memory from SCITAS may result on more queuing time.

## Acknowledgement
We thank Wenjing Bian et. al for their excellent open-source implementation of [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf/) 
and Lihe Yang et. al for their excellent open-source implementation of [Depth-Anything](https://github.com/LiheYoung/Depth-Anything.git) 
from which we based much of our work on.