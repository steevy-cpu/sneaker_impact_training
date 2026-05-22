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
# Preferred: select the camera EXPLICITLY by name so it always picks your
# external USB-C camera, never the built-in webcam, even if index numbers
# change. Set CAMERA_NAME to part of the camera's name (case-insensitive),
# e.g. "USB". Run `python list_cameras.py` to see the exact names.
# Leave it as "" to select by index instead.
CAMERA_NAME = "Logitech"   # external Logitech Webcam C930e (not built-in FaceTime)

# Numeric fallback, used only when CAMERA_NAME is "" or doesn't match.
# 0 is usually the built-in webcam; an external USB-C camera is often 1 or 2.
CAMERA_INDEX = 1

# --- Detection ------------------------------------------------------------
MODEL_PATH = "yolov8n.pt"            # YOLO weights to load (later phases)
CONFIDENCE_THRESHOLD = 0.5           # minimum YOLO confidence to keep a box
MAX_DETECTIONS = 5                   # cap on shoes processed per frame

# --- Output ---------------------------------------------------------------
OUTPUT_ROOT = "sneaker_impact/pictures"   # where crops + metadata will be saved
SAVE_FULL_FRAME = False                   # also save the full frame beside crop

# --- Color detection (future) ---------------------------------------------
ENABLE_COLOR_DETECTION = False

# --- Tracking & UI (future) -----------------------------------------------
TRACK_EXPIRATION_FRAMES = 15         # frames a shoe may be missing before save
DISPLAY_FPS = True                   # draw an FPS overlay on the live feed