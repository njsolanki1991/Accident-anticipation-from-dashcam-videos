"""Inspect a feature .npz in VS Code.

HOW TO USE (VS Code):
  1. Install the "Python" and "Jupyter" extensions.
  2. Click "Run Cell" above a `# %%` block (or press Shift+Enter).
  3. Open the Variables panel in the Interactive window, then click the
     grid/table icon next to a 2-D array (e.g. `video0_slot0`) to open the
     Data Viewer and scroll the actual numbers.

Change NPZ_PATH below to view a different batch.
"""

# %% Load the file ----------------------------------------------------------
import numpy as np

NPZ_PATH = "Dataset/features/testing/batch_001.npz"   # <-- change me if needed

f = np.load(NPZ_PATH)
print("file:", NPZ_PATH)
print("keys:", f.files)
for k in f.files:
    print(f"  {k:8s} shape={f[k].shape} dtype={f[k].dtype}")

data = f["data"]        # (10 videos, 100 frames, 20 slots, 4096 features)
labels = f["labels"]    # (10 videos, 2)  one-hot [no-accident, accident]

# %% Look at the labels ------------------------------------------------------
print("labels (per video):")
for i, lab in enumerate(labels):
    kind = "ACCIDENT" if lab[1] == 1 else "no-accident"
    print(f"  video {i}: {lab.tolist()}  -> {kind}")

# %% 2-D slices you can open in the Data Viewer ------------------------------
# Data Viewer only shows 1-D or 2-D arrays, so we slice the 4-D tensor down.

VIDEO = 0   # which of the 10 videos

# whole-frame feature (slot 0) across all 100 frames -> (100, 4096)
video0_slot0 = data[VIDEO, :, 0, :]

# one frame, all 20 slots -> (20, 4096)  (slot 0 = frame, 1..19 = objects)
video0_frame0 = data[VIDEO, 0, :, :]

print("video0_slot0:", video0_slot0.shape, "  <- right-click -> View in Data Viewer")
print("video0_frame0:", video0_frame0.shape)

# %% How many object slots are actually filled each frame --------------------
# (slot is "filled" if its 4096-vector is not all zeros)
occupied = (np.abs(data[VIDEO]).sum(axis=-1) > 0)   # (100 frames, 20 slots)
print("slots used per frame (first 15 frames):", occupied.sum(axis=1)[:15].tolist())
print("max objects in any frame:", int(occupied[:, 1:].sum(axis=1).max()))
