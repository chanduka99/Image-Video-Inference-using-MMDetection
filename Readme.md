# Object Detection with MMDetection

This project provides a streamlined workflow for running state-of-the-art object detection using the MMDetection toolbox from OpenMMLab. It includes scripts for performing inference on both images and videos, as well as utilities for managing model weights and verifying your environment setup.

**Key Features:**
- **Easy Inference:** Run object detection on images and videos with a single command.
- **Automatic Model Management:** Download and manage compatible model weights automatically.
- **Environment Validation:** Tools to check your MMCV, PyTorch, and CUDA installation for compatibility.
- **Flexible Configuration:** Easily switch between different detection models and thresholds.

Whether you're a researcher, developer, or hobbyist, this project helps you quickly get started with powerful object detection models, ensuring you have the right dependencies and configurations for a smooth experience.



# Installation Guide for MMDetection Setup
#### Note: I have included exact version of almost all the packages installed. They are necessary to maintain comapatability. In case if you are installing newer versions make sure they are compatible with each other.
## Step 1: Create virtual environment
```bash
conda create --prefix ./mmcv
```
```bash
conda activate ./mmcv
```

## Step 2: Install PyTorch 2.1.0 with CUDA 12.1
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```


## Step 2: Install MMCV and MMEngine libraries
```bash
pip install -U openmim
```
```bash
mim install mmcv==2.1.0
```
```bash
mim install mmengine==0.10.7
```


## Step 3: Install MMDetection library
```bash
mim install mmdet==3.3.0
```

## Step 4: Downgrade NumPy to 1.24.4 (required for PyTorch/MMCV compatibility)
```bash
pip install numpy==1.24.4
```
## Step 5: Install TensorBoard
```bash
pip install tensorboard
```

## Step 6: Install a compatible OpenCV version
```bash
pip install opencv-python==4.12.0.88
```


## Step 7: Pin protobuf to avoid runtime errors
```bash
pip install protobuf==3.20.3
```

## Step 8: Clone the mmdetection repository
```bash
git clone https://github.com/open-mmlab/mmdetection.git
```

## Step 9: Verify successfull installation
This repo contains a file called **verify_mmcv_installation.py**. Run it.
```bash
python verify_mmcv_installation.py
```
Screenshot of a successful install is below:
![finding the correct pytorch version and cuda](/install_success.png "Matching pytorch version and cuda")