"""
STEP 1 — Video -> Frames with OpenCV
====================================

Goal: open one dashcam video, understand what a "frame" is, and save one
frame to disk so we can look at it.

Run it from the project root:
    python learning/01_video_to_frames.py

Then open the saved image at:  learning/step1_output/frame_0050.jpg
"""

import os
import cv2   # OpenCV

# ---------------------------------------------------------------------------
# 1. Pick a video to look at.
# ---------------------------------------------------------------------------
VIDEO_PATH = "Dataset/videos/test/Accident/000456.mp4"

# Where we'll save the frame we extract.
OUT_DIR = "learning/step1_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Open the video.
#    VideoCapture is OpenCV's "video reader". It does NOT load the whole video
#    into memory — it lets us pull frames one at a time.
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

# ---------------------------------------------------------------------------
# 3. Ask the video about itself (metadata).
# ---------------------------------------------------------------------------
fps          = cap.get(cv2.CAP_PROP_FPS)
frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("VIDEO PROPERTIES")
print(f"  file        : {VIDEO_PATH}")
print(f"  frames      : {frame_count}")
print(f"  fps         : {fps}")
print(f"  size (WxH)  : {width} x {height}")
print(f"  duration    : {frame_count / fps:.1f} seconds")
print()

# ---------------------------------------------------------------------------
# 4. Read every frame, one at a time, into a list.
#    Each call to cap.read() returns:
#        ok    -> True while there are still frames, False at the end
#        frame -> a NumPy array of shape (height, width, 3), colors in BGR
# ---------------------------------------------------------------------------
frames = []
while True:
    ok, frame = cap.read()
    if not ok:
        break            # no more frames -> stop
    frames.append(frame)

cap.release()            # always release the video reader when done

print("READING RESULT")
print(f"  frames actually read : {len(frames)}")
print(f"  one frame's shape    : {frames[0].shape}   (rows, cols, color-channels)")
print(f"  one frame's dtype    : {frames[0].dtype}   (uint8 = pixel values 0-255)")
print()

# ---------------------------------------------------------------------------
# 5. Save frame #50 so we can SEE a single frame.
#    (cv2.imwrite expects BGR, which is exactly what OpenCV gave us, so no
#     color conversion is needed when saving with OpenCV itself.)
# ---------------------------------------------------------------------------
frame_index = 50
one_frame = frames[frame_index]
out_path = os.path.join(OUT_DIR, f"frame_{frame_index:04d}.jpg")
cv2.imwrite(out_path, one_frame)

print("SAVED A FRAME")
print(f"  saved frame #{frame_index} to: {out_path}")
print("  -> open that file and you'll see one still image from the dashcam.")
