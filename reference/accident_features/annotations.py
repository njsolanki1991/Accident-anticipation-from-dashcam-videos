"""Optional ground-truth object boxes from the annotation .txt files.

Annotation line format (whitespace separated):

    image_name  track_ID  class  x1  y1  x2  y2  accident_flag(0/1)

NOTE: in the copied dataset the annotation files (000001-000010) do NOT
correspond to any of the copied videos, so this source is unused by default.
It exists so that, if you later obtain videos whose names match the annotation
files, you can build features from the author's ground-truth boxes instead of
running a detector -- useful as a faithfulness comparison for the thesis.

``load_boxes`` returns a list of length N_FRAMES; each item is a list of
(x1, y1, x2, y2) boxes for that frame, ordered by track_ID (stable), capped at
N_OBJECTS.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from . import config

Box = Tuple[float, float, float, float]


def load_boxes(annotation_path: str | Path, n_frames: int = config.N_FRAMES) -> List[List[Box]]:
    annotation_path = Path(annotation_path)
    # frame_name -> list of (track_id, box)
    per_frame: dict[str, list[tuple[int, Box]]] = defaultdict(list)
    frame_order: list[str] = []

    with open(annotation_path, "r") as fh:
        for raw in fh:
            parts = raw.split()
            if len(parts) < 7:
                continue
            frame_name = parts[0]
            try:
                track_id = int(parts[1])
                x1, y1, x2, y2 = (float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]))
            except ValueError:
                continue
            if frame_name not in per_frame:
                frame_order.append(frame_name)
            per_frame[frame_name].append((track_id, (x1, y1, x2, y2)))

    # Map the distinct frame names (in file order) to frame indices 0..N-1.
    frames: List[List[Box]] = []
    for frame_name in frame_order[:n_frames]:
        objs = sorted(per_frame[frame_name], key=lambda item: item[0])  # by track_id
        frames.append([box for _, box in objs[: config.N_OBJECTS]])

    # Pad to exactly n_frames with empty box lists if the file was short.
    while len(frames) < n_frames:
        frames.append([])
    return frames
