"""Per-frame object detection with a torchvision Faster R-CNN.

The original paper used Faster R-CNN to propose up to 19 objects per frame;
their detection code was never released. This module reproduces that step with
torchvision's COCO-pretrained ``fasterrcnn_resnet50_fpn``.

Each detection is a box (x1, y1, x2, y2) in pixel coordinates. Boxes are
filtered to vehicle/person classes (see config.COCO_KEEP_LABELS), thresholded
by score, sorted by score descending, and capped at config.N_OBJECTS.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch

from . import config

Box = Tuple[float, float, float, float]


class ObjectDetector:
    def __init__(self, device: str = "cpu", score_thresh: float | None = None):
        self.device = torch.device(device)
        self.score_thresh = (
            config.DETECTION_SCORE_THRESH if score_thresh is None else score_thresh
        )
        self.model = self._load_model().to(self.device).eval()

    @staticmethod
    def _load_model():
        from torchvision.models import detection as det

        # Prefer the modern weights API (torchvision >= 0.13); fall back to the
        # legacy pretrained= flag on older installs.
        try:
            weights = det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            return det.fasterrcnn_resnet50_fpn(weights=weights)
        except AttributeError:  # pragma: no cover - very old torchvision
            return det.fasterrcnn_resnet50_fpn(pretrained=True)

    @torch.no_grad()
    def detect(self, frame_rgb_float: torch.Tensor) -> List[Box]:
        """Detect objects in one frame.

        ``frame_rgb_float`` is a (3, H, W) float tensor in [0, 1], RGB order
        (Faster R-CNN does its own normalization internally).
        Returns up to N_OBJECTS boxes, highest score first.
        """
        out = self.model([frame_rgb_float.to(self.device)])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        keep = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.score_thresh:
                continue
            if int(label) not in config.COCO_KEEP_LABELS:
                continue
            keep.append((float(score), tuple(float(v) for v in box)))

        # Highest score first, then cap at the number of object slots.
        keep.sort(key=lambda item: item[0], reverse=True)
        return [box for _, box in keep[: config.N_OBJECTS]]

    def detect_bgr(self, frame_bgr: np.ndarray) -> List[Box]:
        """Convenience: detect objects from a BGR uint8 frame (as cv2 returns)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return self.detect(tensor)
