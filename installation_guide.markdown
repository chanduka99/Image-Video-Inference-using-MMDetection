# Installation Guide for PyTorch and MMDetection Setup

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