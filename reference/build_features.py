#!/usr/bin/env python
"""Build DSA-RNN feature files (batch_XXX.npz) from dashcam videos.

This reproduces the Anticipating-Accidents author's unreleased video -> features
step. Output goes to Dataset/features_generated/{training,testing}/ so the
author's reference files in Dataset/features/ are never overwritten.

Examples
--------
Smoke test on 2 videos (CPU):
    python build_features.py --split test --limit 2

Full test split with the detector:
    python build_features.py --split test

Both splits:
    python build_features.py --split all

Use ground-truth annotation boxes instead of the detector (only for videos that
have a matching Dataset/annotation/<name>.txt):
    python build_features.py --split train --object-source annotation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from accident_features import config
from accident_features.builder import build_split
from accident_features.detector import ObjectDetector
from accident_features.vgg_features import VGGFeatureExtractor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "test", "all"], default="test",
                   help="Which split to process (default: test).")
    p.add_argument("--object-source", choices=["detector", "annotation"], default="detector",
                   help="Where object boxes come from (default: detector).")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda (default: cpu).")
    p.add_argument("--score-thresh", type=float, default=config.DETECTION_SCORE_THRESH,
                   help="Detection score threshold (default: %(default)s).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N videos of the split (for quick tests).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output root (default: Dataset/features_generated).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("Loading VGG-16 feature extractor ...")
    extractor = VGGFeatureExtractor(device=args.device)

    detector = None
    if args.object_source == "detector":
        print("Loading Faster R-CNN detector ...")
        detector = ObjectDetector(device=args.device, score_thresh=args.score_thresh)

    splits = ["train", "test"] if args.split == "all" else [args.split]
    t0 = time.time()
    for split in splits:
        print(f"\n=== Building split: {split} ===")
        build_split(
            split,
            extractor=extractor,
            detector=detector,
            object_source=args.object_source,
            limit=args.limit,
            out_root=args.out_dir,
        )
    print(f"\nDone in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
