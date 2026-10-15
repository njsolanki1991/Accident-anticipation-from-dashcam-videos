"""Orchestration: videos -> (100, 20, 4096) feature tensors -> batch_XXX.npz.

Pipeline per video:
  1. read exactly N_FRAMES frames
  2. per frame: take the whole frame (slot 0) plus up to N_OBJECTS object crops
     (slots 1..19) -- object boxes come from the detector (default) or from the
     ground-truth annotation file
  3. run every crop through VGG-16 fc7 -> 4096-d features
  4. assemble (N_FRAMES, N_DETECTION, FEAT_DIM)

Videos are grouped BATCH_SIZE at a time and written as float16 .npz files whose
keys match what accident.py reads: ``data`` and ``labels``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import annotations as ann
from . import config, video_io
from .detector import ObjectDetector
from .vgg_features import VGGFeatureExtractor

VideoItem = Tuple[Path, Tuple[float, float]]  # (video_path, one-hot label)


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def _crop(frame: np.ndarray, box) -> Optional[np.ndarray]:
    """Return a valid BGR crop for ``box`` (x1,y1,x2,y2), or None if degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


# --------------------------------------------------------------------------- #
# Per-video feature extraction
# --------------------------------------------------------------------------- #
def extract_video_features(
    video_path: Path,
    extractor: VGGFeatureExtractor,
    detector: Optional[ObjectDetector] = None,
    annotation_path: Optional[Path] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Return a (N_FRAMES, N_DETECTION, FEAT_DIM) float32 feature tensor."""
    if detector is None and annotation_path is None:
        raise ValueError("Provide either a detector or an annotation_path for object boxes.")

    frames = video_io.read_frames(video_path)
    gt_boxes = ann.load_boxes(annotation_path) if annotation_path is not None else None

    out = np.zeros((config.N_FRAMES, config.N_DETECTION, config.FEAT_DIM), dtype=np.float32)

    for t, frame in enumerate(frames):
        # Object boxes for this frame.
        if gt_boxes is not None:
            boxes = gt_boxes[t]
        else:
            boxes = detector.detect_bgr(frame)

        crops = [frame]  # slot 0 == whole frame
        for box in boxes:
            crop = _crop(frame, box)
            if crop is not None:
                crops.append(crop)
            if len(crops) == config.N_DETECTION:  # 1 frame + 19 objects
                break

        feats = extractor.extract(crops)  # (len(crops), 4096)
        out[t, : feats.shape[0], :] = feats

        if verbose and (t == 0 or (t + 1) % 25 == 0):
            print(f"      frame {t + 1:3d}/{config.N_FRAMES}  slots used: {feats.shape[0]}")

    return out


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def discover_videos(split: str) -> List[VideoItem]:
    """List (video_path, label) pairs for a split ('train' or 'test')."""
    subdir = config.SPLITS[split]["videos_subdir"]
    base = config.VIDEOS_DIR / subdir
    items: List[VideoItem] = []
    for folder, label in config.LABEL_FOLDERS.items():
        folder_dir = base / folder
        if not folder_dir.is_dir():
            continue
        for video in sorted(folder_dir.glob("*.mp4")):
            items.append((video, label))
    return items


# --------------------------------------------------------------------------- #
# Batch writing
# --------------------------------------------------------------------------- #
def _write_batch(data: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        data=data.astype(config.FEATURE_DTYPE),
        labels=labels.astype(config.FEATURE_DTYPE),
    )


def build_split(
    split: str,
    extractor: VGGFeatureExtractor,
    detector: Optional[ObjectDetector],
    object_source: str = "detector",
    limit: Optional[int] = None,
    start_batch: int = 1,
    out_root: Optional[Path] = None,
) -> List[Path]:
    """Build all batch_XXX.npz files for one split. Returns written paths.

    Partial final batches (fewer than BATCH_SIZE videos) are still written so no
    data is lost; note accident.py expects exactly BATCH_SIZE videos per file,
    so for training you normally want a multiple of BATCH_SIZE videos.
    """
    items = discover_videos(split)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise RuntimeError(f"No videos found for split '{split}' under {config.VIDEOS_DIR}")

    if out_root is None:
        out_root = config.GENERATED_FEATURES_DIR
    out_dir = out_root / config.SPLITS[split]["out"]
    written: List[Path] = []
    batch_idx = start_batch

    for start in range(0, len(items), config.BATCH_SIZE):
        chunk = items[start : start + config.BATCH_SIZE]
        n = len(chunk)
        data = np.zeros((n, config.N_FRAMES, config.N_DETECTION, config.FEAT_DIM), dtype=np.float32)
        labels = np.zeros((n, config.N_CLASSES), dtype=np.float32)

        for i, (video_path, label) in enumerate(chunk):
            print(f"  [{split}] batch {batch_idx:03d}  video {i + 1}/{n}: {video_path.name}")
            annotation_path = None
            if object_source == "annotation":
                annotation_path = config.ANNOTATION_DIR / (video_path.stem + ".txt")
                if not annotation_path.exists():
                    raise FileNotFoundError(
                        f"No annotation for {video_path.name} at {annotation_path}. "
                        "Use --object-source detector for these videos."
                    )
            data[i] = extract_video_features(
                video_path,
                extractor=extractor,
                detector=None if object_source == "annotation" else detector,
                annotation_path=annotation_path,
                verbose=True,
            )
            labels[i] = label

        out_path = out_dir / f"batch_{batch_idx:03d}.npz"
        _write_batch(data, labels, out_path)
        print(f"  -> wrote {out_path}  data={data.shape} labels={labels.shape}")
        written.append(out_path)
        batch_idx += 1

    return written
