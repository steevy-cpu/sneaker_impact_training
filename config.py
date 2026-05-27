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
# Two model families are supported:
#  - USE_YOLO_WORLD=True: YOLO-World (open-vocabulary), prompted with the
#    classes in YOLO_WORLD_CLASSES. Better at uncommon shoe types (five-toe
#    shoes, etc.) because it understands text labels, not just a single
#    "Footwear" bucket. Small variant chosen for Pi 5 / Jetson Nano viability.
#  - USE_YOLO_WORLD=False: standard YOLO with MODEL_PATH (default OIV7 medium).
USE_YOLO_WORLD = True
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"   # ~28MB, runs on Pi 5 / Jetson Nano
YOLO_WORLD_CLASSES = [                    # prompts -- tune freely
    "shoe", "sneaker", "running shoe", "boot", "sandal",
    "flip flop", "high heel", "toe shoe", "athletic shoe", "loafer",
]

MODEL_PATH = "yolov8m-oiv7.pt"       # used when USE_YOLO_WORLD=False
CONFIDENCE_THRESHOLD = 0.5           # minimum YOLO confidence to keep a box
MAX_DETECTIONS = 5                   # cap on shoes processed per frame
YOLO_IMGSZ = 416                     # YOLO input size; default ultralytics
                                     # uses 640. 416 is ~1.5x faster with a
                                     # small accuracy hit; 320 even faster.
YOLO_DEVICE = "auto"                 # "auto" (MPS on Apple Silicon else CPU),
                                     # or pin to "cpu" / "mps" / "cuda:0"
ENABLE_GRABCUT = True                # run GrabCut on each bbox to get a
                                     # shoe-shaped polygon for the live mask;
                                     # set False if it's too slow/noisy
GRABCUT_ITERS = 1                    # GrabCut iterations; 1 is usually enough,
                                     # 3 is the OpenCV default but ~3x slower
GRABCUT_REFRESH_CYCLES = 8           # reuse a cached polygon for up to this
                                     # many detector cycles for a stable shoe;
                                     # higher = faster, less responsive to
                                     # shape changes (rotations, deformation)

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
MASK_SHRINK = 0.7                    # draw mask at this fraction of bbox size,
                                     # centered, so adjacent shoes stay visible
                                     # (click target still uses the full bbox)
DISPLAY_FPS = True                   # draw an FPS overlay on the live feed