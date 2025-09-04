# Installation Guide for PyTorch and MMDetection Setup

 **check the latest mmcv version >> find matching Pytorch version >> find matching cuda version**.

first go to the mmcv releases page and see the needed pytorch version for the latest released mmcv version

below is the screenshot from the release page.

![finding the correct pytorch version for the mmcv version](/mmcv_pytorch_version.png "Matching mmcv version and pytorch version")

Now we know for the latest mmcv 2.10 we need pytorch 2.1.0.

lets go to the pytorch websiste and see what are the matching cuda version availble for pytorch 2.1.0.
![finding the correct pytorch version and cuda](/pytorch_cuda_1.png "Matching pytorch version and cuda")
currently pytorch 2.8.0 is available that are compatible with cuda 12.6, 12.8 and 12.9 versions.  

But for pytorch 2.8.0 version there are no any matching mmcv pre-built packages. Of course you are free to built from source. Still why go through that trouble?

Therefore we will go to the link **"install previous versions of Pytorch".**

After scrollling down a little we find this:

![finding the correct pytorch version and cuda](/pytorch_cuda_2.png "Matching pytorch version and cuda")

For **Pytorch 2.1.0** there are two matching cuda versions as **cuda 11.8** and **cuda 12.1**

### Now for we found that for **mmcv v2.1.0** the matching **Pytorch v2.1.0** and for Pytorch the matching cuda versions are **cuda 11.8** or **cuda 12.1**. This is the most crucial step for setting up mmdetection locally.

### ----------------------------Optional ----------------------------
You can cross check if there are prebuilt mmcv versions for specific Pytorch and cuda versions.

For example:

if you go to the following link. you can see the existing mmcv versions for cuda 12.1 and 2.1.0.

[https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html)

use the following pattern in the url to see the if there are mmcv versions available for the specific {cuda version} and {pytorch version}

```
https://download.openmmlab.com/mmcv/dist/cu{cuda version}/torch{pytorch version}/index.html
```
It's better to always first **check the latest mmcv version >> find matching Pytorch version >> find matching cuda version**. It will make your life easy. Trust me I gave gone the other way round and spend more time working on fixing compatability issues rather than the actual project :(.

### ------------------------------------------------------------------




## Step 1: Install PyTorch 1.11.0 with CUDA 11.3
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Step 2: Install MMCV-Full 1.7.2 (matching PyTorch 1.11.0 + CUDA 11.3)
```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

## Step 3: Install MMDetection 2.28.2
```bash
pip install mmdet==2.28.2
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
pip install opencv-python==4.5.5.64
```

## Step 7: Pin protobuf to avoid runtime errors
```bash
pip install protobuf==3.20.3
```