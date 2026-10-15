"""Feature-creation pipeline for the Anticipating-Accidents (DSA-RNN) model.

Turns raw dashcam videos into the ``batch_XXX.npz`` feature files that
``accident.py`` consumes. This reproduces the author's unreleased
video -> features step.

See README_feature_pipeline.md for the full description and caveats.
"""

__all__ = [
    "config",
    "video_io",
    "detector",
    "vgg_features",
    "annotations",
    "builder",
]
