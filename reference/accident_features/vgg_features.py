"""VGG-16 fc7 feature extraction (4096-d, post-ReLU).

The original features are the 4096-d VGG fully-connected activations. Inspecting
the author's npz files showed all values are >= 0, i.e. the activations are
taken AFTER the ReLU. We reproduce that by running crops through VGG-16 up to
and including the second ReLU in the classifier head (the "fc7 relu" output).

Note on faithfulness: the author used a Caffe VGG; here we use torchvision's
ImageNet-pretrained VGG-16. The feature *schema* (4096-d, post-ReLU, non-negative)
is identical, but the exact numbers differ because the weights differ. See the
README for what this means for using the author's pretrained demo_model.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import config


class VGGFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.features, self.avgpool, self.fc7 = self._build()
        self._mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)

    def _build(self):
        from torchvision.models import vgg16

        try:
            from torchvision.models import VGG16_Weights

            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):  # pragma: no cover - old torchvision
            model = vgg16(pretrained=True)

        model.eval()
        # classifier = [Linear(25088,4096), ReLU, Dropout, Linear(4096,4096),
        #               ReLU, Dropout, Linear(4096,1000)]
        # Indices [:5] -> output of the second ReLU == fc7 post-ReLU (4096-d).
        fc7 = nn.Sequential(*list(model.classifier.children())[:5])
        features = model.features.to(self.device).eval()
        avgpool = model.avgpool.to(self.device).eval()
        fc7 = fc7.to(self.device).eval()
        return features, avgpool, fc7

    def _preprocess(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        """BGR uint8 crops (any size) -> normalized (N, 3, 224, 224) tensor."""
        batch = np.empty((len(crops_bgr), config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            batch[i] = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)  # (N,3,H,W)
        tensor = (tensor - self._mean) / self._std
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Return (N, 4096) float32 features for a list of BGR crops."""
        if not crops_bgr:
            return np.zeros((0, config.FEAT_DIM), dtype=np.float32)
        x = self._preprocess(crops_bgr)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc7(x)  # (N, 4096), post-ReLU (>= 0)
        return x.cpu().numpy().astype(np.float32)
