"""Read a video into exactly N_FRAMES frames.

The copied dataset clips are already exactly 100 frames, but this helper is
robust to clips with a different frame count: it samples uniformly when there
are too many frames and repeats the last frame when there are too few.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from . import config


def read_frames(video_path: str | Path, n_frames: int = config.N_FRAMES) -> List[np.ndarray]:
    """Return a list of exactly ``n_frames`` BGR frames (uint8 HxWx3).

    Raises FileNotFoundError if the video cannot be opened or has no frames.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise FileNotFoundError(f"No frames decoded from video: {video_path}")

    return _resample(frames, n_frames)


def _resample(frames: List[np.ndarray], n_frames: int) -> List[np.ndarray]:
    count = len(frames)
    if count == n_frames:
        return frames
    if count > n_frames:
        # Uniformly spaced indices across the clip.
        idx = np.linspace(0, count - 1, n_frames).round().astype(int)
        return [frames[i] for i in idx]
    # Too few frames: pad by repeating the final frame.
    return frames + [frames[-1]] * (n_frames - count)
