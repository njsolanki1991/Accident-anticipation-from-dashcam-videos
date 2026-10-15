"""Central configuration and constants for the feature pipeline.

Every shape here is dictated by the original model (accident.py) and was
verified against the author's precomputed batch_XXX.npz files:

    data   : (BATCH_SIZE, N_FRAMES, N_DETECTION, FEAT_DIM)  float16, values >= 0
    labels : (BATCH_SIZE, 2)                                float16, one-hot

Slot 0 of the N_DETECTION axis is the whole-frame feature; slots 1..19 hold
up to N_OBJECTS detected objects, packed from the lowest index upward. Unused
object slots stay all-zero (the model masks them out).
"""

from __future__ import annotations

from pathlib import Path

# --- Tensor shape constants (do not change: fixed by accident.py) ----------
N_FRAMES = 100          # time steps per clip
N_DETECTION = 20        # slot 0 = whole frame, slots 1..19 = objects
N_OBJECTS = N_DETECTION - 1  # == 19 object slots
FEAT_DIM = 4096         # VGG-16 fc7 length
N_CLASSES = 2           # one-hot [no-accident, accident]
BATCH_SIZE = 10         # videos packed per .npz file
FEATURE_DTYPE = "float16"  # author stores features as float16

# One-hot label vectors (matches accident.py: labels[i][1] == 1 => accident)
LABEL_NO_ACCIDENT = (1.0, 0.0)
LABEL_ACCIDENT = (0.0, 1.0)

# --- Image preprocessing (ImageNet / VGG-16) -------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Object detection ------------------------------------------------------
# COCO category ids (torchvision Faster R-CNN) kept as "objects" of interest
# for dashcam accident anticipation.
COCO_KEEP_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    7: "train",
    8: "truck",
}
DETECTION_SCORE_THRESH = 0.5  # keep detections with score >= this

# --- Default dataset layout (relative to the project root) -----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
FEATURES_DIR = DATASET_DIR / "features"              # author's reference .npz (read-only)
GENERATED_FEATURES_DIR = DATASET_DIR / "features_generated"  # our output (never clobbers the above)
ANNOTATION_DIR = DATASET_DIR / "annotation"

# Folder name -> (split output dir name, video subfolders with their label)
# Matches the author's directory names: features/{training,testing}/
SPLITS = {
    "train": {
        "out": "training",
        "videos_subdir": "train",
    },
    "test": {
        "out": "testing",
        "videos_subdir": "test",
    },
}

# Video label subfolders -> one-hot label
LABEL_FOLDERS = {
    "Accident": LABEL_ACCIDENT,
    "NonAccident": LABEL_NO_ACCIDENT,
}
