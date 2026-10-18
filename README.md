# Dashcam Accident Anticipation — Feature-Creation Pipeline

Master's thesis project (University of Passau, MES) building on
**"Anticipating Accidents in Dashcam Videos"** (Chan et al., ACCV 2016) and its
reference model [smallcorgi/Anticipating-Accidents](https://github.com/smallcorgi/Anticipating-Accidents).

## What this project is

The original paper predicts a traffic accident **before it happens** from a
dashcam clip, using a **Dynamic-Spatial-Attention RNN (DSA-RNN)**. The authors
released the model (`accident.py`) and precomputed CNN features, but **not** the
code that turns raw video into those features.

**Contribution of this repo:** a from-scratch reimplementation of that missing
**video → features** step — reverse-engineered from the authors' own data
format — plus a step-by-step notebook walkthrough explaining every stage.

## The two-stage pipeline

```
STAGE A (this repo):  video ──► object detection ──► CNN features ──► batch_XXX.npz
STAGE B (authors'):   batch_XXX.npz ──► DSA-RNN ──► per-frame accident probability
```

Each `.npz` holds one batch of 10 videos:
- `data`   : `(10, 100, 20, 4096)` float16 — 10 videos × 100 frames × 20 slots × 4096-d features
  (slot 0 = whole frame, slots 1–19 = detected objects)
- `labels` : `(10, 2)` float16 — one-hot `[no-accident, accident]`

Features are **VGG-16 fc7** (4096-d, post-ReLU); objects are found with a
**Faster R-CNN** detector.

## Repository layout

```
learning/     step-by-step Jupyter notebooks (the walkthrough)
  01_video_to_frames.ipynb   video → frames (OpenCV)
  02_detect_objects.ipynb    detect objects (Faster R-CNN)
  03_vgg_features.ipynb      crop → 4096 features (VGG-16)
  04_frame_to_slots.ipynb    one frame → (20, 4096)
  05_video_to_tensor.ipynb   one video → (100, 20, 4096)
  06_build_npz.ipynb         videos + labels → batch_XXX.npz (5 accident + 5 non)
  06_build_npz_kaggle.ipynb  GPU version for unattended runs on Kaggle
  view_npz.ipynb             inspect any .npz batch and its labels
dataset.txt   dataset layout notes
```

## Setup

```bash
pip install -r requirements.txt
```

Confirmed environment: Python 3.9, NumPy 1.26, OpenCV 4.10, PyTorch 1.13 (CPU),
torchvision 0.14. The Faster R-CNN and VGG-16 weights download automatically on
first use.

## The dataset (not included)

The ~1,740 dashcam clips and the authors' precomputed features are **not** in
this repo (tens of GB). Download them from the original project and arrange as:

```
Dataset/
├─ videos/{train,test}/{Accident,NonAccident}/*.mp4
├─ features/{train,test}/batch_XXX.npz   (authors' reference features)
└─ annotation/*.txt                            (accident videos only)
```

- Accident videos: 610 (train 11–455, test 456–620)
- NonAccident videos: 1130 (train 1–829, test 830–1130)
- Annotations exist for accident videos only.

## How to run

Open the notebooks in `learning/` and run them in order (01 → 06). Notebook 06
produces a `batch_001.npz` under `Dataset/features_generated/` — the authors'
reference files are never overwritten.

## Citation

Chan, F.-H., Chen, Y.-T., Xiang, Y., & Sun, M. (2016). *Anticipating Accidents
in Dashcam Videos.* Asian Conference on Computer Vision (ACCV), 136–153.

## Notes

This repo implements the **feature-creation** stage. The DSA-RNN model itself is
the original authors' work (TensorFlow 1.x); see their repository.
