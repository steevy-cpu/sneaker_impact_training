"""
Central configuration for the Sneaker Impact shoe data-collection system.

PURPOSE
-------
Every tunable value for the project lives here so that no script ever
hardcodes a camera index, file path, or threshold. Other modules will
`import config` and read these constants.

NOTE
----
As of Phase 1 the camera layer is implemented (see camera_utils.py). Detection,
tracking, saving, and color logic are NOT implemented yet -- later phases.
"""

# --- Camera ---------------------------------------------------------------
# Pick the camera by NUMERIC INDEX. To find it: run `python list_cameras.py`,
# which probes 0..5 and saves a preview JPG from each working index to /tmp/
# so you can visually identify which one is your external camera.
#
# macOS quirk: name-based lookup is unreliable. macOS `system_profiler` lists
# cameras in one order, but OpenCV's AVFoundation backend may use a different
# order, so picking by name can silently grab the wrong camera. Use the index.
CAMERA_NAME = ""           # leave empty; name lookup isn't reliable on macOS
CAMERA_INDEX = 0           # OpenCV index for the Logitech Webcam C930e here

# --- Detection ------------------------------------------------------------
MODEL_PATH = "yolov8m-oiv7.pt"       # Open Images V7 -- has "Footwear" class
CONFIDENCE_THRESHOLD = 0.5           # minimum YOLO confidence to keep a box
MAX_DETECTIONS = 5                   # cap on shoes processed per frame

# --- Output ---------------------------------------------------------------
OUTPUT_ROOT = "sneaker_impact/pictures"   # where crops + metadata will be saved
SAVE_FULL_FRAME = False                   # also save the full frame beside crop

# --- Color detection (future) ---------------------------------------------
ENABLE_COLOR_DETECTION = False

# --- Tracking & UI --------------------------------------------------------
TRACK_EXPIRATION_FRAMES = 60         # frames a shoe may be missing before save
                                     # (~2-6s depending on detection FPS; raise
                                     # this if shoes auto-save as Reuse before
                                     # the operator has time to double-click)
TRACK_IOU_THRESHOLD = 0.3            # min IoU to match a detection to a track
DISPLAY_FPS = True                   # draw an FPS overlay on the live feed