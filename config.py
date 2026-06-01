"""
Central configuration for the Sneaker Impact shoe data-collection system.

PURPOSE
-------
Every tunable value for the project lives here so that no script ever
hardcodes a camera index, file path, or threshold. Other modules will
`import config` and read these constants.

NOTE
----
Phases 1-6 are implemented: camera, live detection/labeling, tracking, dataset
storage, color detection, and the dataset quality tools all read their settings
from here. Keep this file the single source of truth -- no module should
hardcode a camera index, path, or threshold.
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
CAPTURE_WIDTH = 1280       # request this capture resolution from the camera.
CAPTURE_HEIGHT = 720       # Smaller = faster per-frame work everywhere (copies,
                           # YOLO, color, saves) AND smaller saved crops. The
                           # camera snaps to its nearest supported mode;
                           # open_camera() prints the resolution it actually got.
                           # Set BOTH to 0 to keep the camera's native resolution.

# --- Detection ------------------------------------------------------------
# Two model families are supported:
#  - USE_YOLO_WORLD=True: YOLO-World (open-vocabulary), prompted with the
#    classes in YOLO_WORLD_CLASSES. Better at uncommon shoe types (five-toe
#    shoes, etc.) because it understands text labels, not just a single
#    "Footwear" bucket. Small variant chosen for Pi 5 / Jetson Nano viability.
#  - USE_YOLO_WORLD=False: standard YOLO with MODEL_PATH (default OIV7 medium).
USE_YOLO_WORLD = False
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"   # ~28MB, runs on Pi 5 / Jetson Nano
YOLO_WORLD_CLASSES = [                    # prompts -- tune freely
    "shoe", "sneaker", "running shoe", "boot", "sandal",
    "flip flop", "high heel", "toe shoe", "athletic shoe", "loafer",
]

MODEL_PATH = "yolov8m-oiv7.onnx"     # used when USE_YOLO_WORLD=False. Medium is
                                     # the reliable default: nano (yolov8n-oiv7)
                                     # was ~3x faster on Pi 5 but dropped live
                                     # detections too often (shoes flickered out
                                     # while still in frame), and this project
                                     # values labeling accuracy over FPS. The
                                     # .onnx export runs much faster than the .pt
                                     # on Pi 5 CPU. To retry nano, set
                                     # MODEL_PATH = "yolov8n-oiv7.pt".
CONFIDENCE_THRESHOLD = 0.5           # minimum YOLO confidence to keep a box
MAX_DETECTIONS = 5                   # cap on shoes processed per frame
MIN_BBOX_AREA_FRAC = 0               # ignore detections smaller than this
                                     # fraction of the frame area (~60x60 px at
                                     # 720p). Tiny/distant shoes make poor
                                     # training crops. 0 = keep all sizes.
YOLO_IMGSZ = 320                     # YOLO input size; default ultralytics
                                     # uses 640. 416 is ~1.5x faster with a
                                     # small accuracy hit; 320 even faster.
YOLO_DEVICE = "auto"                 # "auto" (MPS on Apple Silicon else CPU),
                                     # or pin to "cpu" / "mps" / "cuda:0"
ENABLE_GRABCUT = False                # run GrabCut on each bbox to get a
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
BLUR_SAVE_FLOOR = 0                       # if > 0, skip auto-saving a Reuse shoe
                                          # whose sharpest crop scores below this
                                          # (variance-of-Laplacian). Operator
                                          # Recycle clicks are ALWAYS saved.
                                          # Pick a value from the "Blur:" numbers
                                          # in dataset_review.py (~half a typical
                                          # sharp value). 0 = disabled.

# --- Color detection ------------------------------------------------------
ENABLE_COLOR_DETECTION = True        # fill detected_color + color_confidence
                                     # on each saved shoe JSON
COLOR_CENTER_FRAC = 0.6              # with no shoe polygon (GrabCut off), sample
                                     # only this centered fraction of the crop
                                     # for color, so background near the bbox
                                     # edges doesn't bias it. 1.0 = whole crop.
COLOR_AMBIGUOUS_MARGIN = 0.10        # if the top color leads the runner-up by
                                     # less than this fraction of pixels, label
                                     # the shoe "multi" rather than guess one.
                                     # 0 = always pick a single winner.
# HSV thresholds (OpenCV convention: H 0-179, S/V 0-255). Tune if neutral or
# warm-tinted shoes land in the wrong bucket.
COLOR_V_BLACK = 50                   # value below this -> black
COLOR_V_WHITE = 180                  # value above this + low saturation -> white
COLOR_S_GRAY = 50                    # saturation below this -> black/gray/white
COLOR_V_BROWN = 200                  # value below this + reddish hue -> brown

# --- Tracking & UI --------------------------------------------------------
TRACK_EXPIRATION_FRAMES = 60         # frames a shoe may be missing before save
                                     # (~2-6s depending on detection FPS; raise
                                     # this if shoes auto-save as Reuse before
                                     # the operator has time to click)
TRACK_IOU_THRESHOLD = 0.3            # min IoU to match a detection to a track
SHARPNESS_RECHECK_MIN_MOVE = 8       # px a shoe's bbox center must move before
                                     # we recompute its sharpness. A still shoe's
                                     # sharpness barely changes, so this skips
                                     # redundant Laplacian work each frame.
                                     # 0 = always recompute (old behavior).
MASK_SHRINK = 0.7                    # draw mask at this fraction of bbox size,
                                     # centered, so adjacent shoes stay visible
                                     # (click target still uses the full bbox)
DISPLAY_FPS = True                   # draw an FPS overlay on the live feed