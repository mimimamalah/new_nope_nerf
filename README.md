# UNKNOWN OBJECT EXTRACTION FROM MULTI-VIEW IMAGES ON SYNTHETIC IMAGES WITH NO POSE PRIOR

**[Data]() | [PDF]() | [Pretrained Models]()**

## Table of Content
- [UNKNOWN OBJECT EXTRACTION FROM MULTI-VIEW IMAGES ON SYNTHETIC IMAGES WITH NO POSE PRIOR](#unknown-object-extraction-from-multi-view-images-on-synthetic-images-with-no-pose-prior)
	- [Table of Content](#table-of-content)
	- [Overview](#overview)
	- [Installation](#installation)
	- [Data](#data)
	- [Monocular Depth Estimation](#monocular-depth-estimation)
	- [Training](#training)
	- [Evaluation](#evaluation)
	- [More Visualisations](#more-visualisations)
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
cd nope-nerf
python −m venv nope_nerf_env source nope_nerf_env
source nope_nerf_env/bin/activate
source env.sh
pip install -r requirement.txt
```

Using Cuda :
```
git clone https://github.com/mimimamalah/new_nope_nerf.git
cd nope-nerf
conda env create -f environment.yaml
conda activate nope-nerf
pip install -r requirement.txt
```

## Data
1. [Ignatius Dataset]():
We used on the Ignatius dataset

2. [Stove Dataset]():

3. [Hot-Dog Dataset]():

4. [V-KITTI]():

5. [Hubble Dataset]():

6. If you want to use your own image sequence with customised camera intrinsics, you need to add an `data/your_folder/your_scene/intrinsics.npz` file to the scene directory. One example of the config file is `configs/your_folder/your_scene/images.yaml` (please add your own data to the `data/your_folder/your_scene/images` directory). 


## Monocular Depth Estimation
If you want to use Depth-Anything




If you want to use the original DPT 
Monocular depth map generation: you can first download the pre-trained DPT model from [this link](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing) provided by [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) to `DPT` directory, then run
```
python preprocess/dpt_depth.py configs/preprocess.yaml
```
to generate monocular depth maps. You need to modify the `cfg['dataloading']['path']` and `cfg['dataloading']['scene']` in `configs/preprocess.yaml` to your own image sequence.

## Training

1. Train a new model from scratch:

```
python train.py configs/Tanks/Ignatius.yaml
```
where you can replace `configs/Tanks/Ignatius.yaml` with other config files.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ./out --port 6006
```

For available training options, please take a look at `configs/default.yaml`.
## Evaluation
1. Evaluate image quality and depth:
```
python evaluation/eval.py configs/Tanks/Ignatius.yaml
```
To evaluate depth: add `--depth` . Note that you need to add ground truth depth maps by yourself.

2. Evaluate poses:
```
python evaluation/eval_poses.py configs/Tanks/Ignatius.yaml
```
To visualise estimated & ground truth trajectories: add `--vis` 


## More Visualisations
Novel view synthesis
```
python vis/render.py configs/Tanks/Ignatius.yaml
```
Pose visualisation (estimated trajectory only)
```
python vis/vis_poses.py configs/Tanks/Ignatius.yaml
```

## Acknowledgement
We thank Wenjing Bian et. al for their excellent open-source implementation of [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf/) 

from which we based much of our work on.