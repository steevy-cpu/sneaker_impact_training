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
DISK_SPACE_WARN_MB = 500                  # warn (startup print + on-screen
                                          # banner + log) when free space on the
                                          # output drive drops below this many MB,
                                          # so an unattended station doesn't
                                          # silently fail to save. 0 = disable.

# --- Color detection ------------------------------------------------------
ENABLE_COLOR_DETECTION = True        # fill detected_color + color_confidence
                                     # on each saved shoe JSON
COLOR_CENTER_FRAC = 0.6              # with no shoe polygon (GrabCut off), sample
                                     # only this centered fraction of the crop
                                     # for color, so background near the bbox
                                     # edges doesn't bias it. 1.0 = whole crop.
# Color naming uses CIELAB ("Lab"), a perceptual color space where distance
# matches how different two colors LOOK to the human eye (better than HSV, and
# much better than raw RGB which mixes brightness into every channel). Neutral
# pixels (black/gray/white) are decided by lightness + chroma; colored pixels by
# nearest Lab anchor. There is no "multi" -- we always keep the single most
# common color (the one with the most spread across the shoe).
COLOR_LAB_CHROMA_MIN = 12            # below this chroma (C* = sqrt(a*^2+b*^2)) a
                                     # pixel is treated as neutral -> black/gray/
                                     # white; above it, it has a real hue.
COLOR_LAB_L_BLACK = 30               # neutral pixel with L* (0-100) below this
                                     # -> black; above COLOR_LAB_L_WHITE -> white;
                                     # in between -> gray.
COLOR_LAB_L_WHITE = 80

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

# --- Logging --------------------------------------------------------------
LOG_TO_FILE = True           # mirror label_live's console output to a
                             # timestamped file in LOG_DIR -- useful for an
                             # unattended capture station. False = console only.
LOG_DIR = "logs"             # where run logs are written (git-ignored)

# --- Dashboard integration ------------------------------------------------
# Push collected shoes to the Sneaker Impact Dashboard (run it in APP_MODE=actual
# on the SAME machine, so crops can be copied into its images/ folder). The crop
# becomes img_top; the operator's Reuse/Recycle label is mirrored into both
# ai_prediction and final_decision (review_status=COMPLETED). dashboard_sync.py
# backfills collected folders; dashboard_client.py does the actual POST.
DASHBOARD_URL = "http://localhost:8000"         # dashboard FastAPI base URL
DASHBOARD_IMAGES_DIR = "sneaker-impact-dash/images"     # the dashboard's images/
                                                # folder on THIS machine; crops are
                                                # copied here and served at /images/.
OPERATOR_ID = "OP-LIVE"                         # recorded on every pushed record
DASHBOARD_PUSH_LIVE = False                     # Phase 2: live push from label_live

# --- Table segmentation (2026 pivot, Phase A) -----------------------------
# New direction: photograph the WHOLE TABLE of shoes, then in the background
# segment it into individual pairs, crop each pair, and (later phases) identify
# make + model. The segmenter backend is swappable so the licensing call can be
# made later WITHOUT touching callers:
#   "yoloe" -- YOLOE-26 open-vocabulary segmentation. Prompt it with text
#              (SEGMENT_PROMPTS) and it segments those, no training needed.
#              AGPL-3.0 -- fine while internal; needs Enterprise license OR a
#              switch to sam2 if this ever ships as a product.
#   "sam2"  -- Segment Anything 2 (Apache-2.0, commercial-safe). Class-agnostic:
#              returns every object mask; callers filter. Future / product path.
# NOTE: plain yolo26-seg is COCO-only (no shoe class), so open-vocab (yoloe) or
# a custom-trained seg model is required to find shoes with zero training.
SEGMENT_BACKEND = "yoloe"               # "yoloe" (AGPL) or "sam2" (Apache)
SEGMENT_MODEL = "yoloe-26s-seg.pt"      # weights for the chosen backend. If the
                                        # YOLOE-26 weights aren't published yet
                                        # under this name, "yoloe-11s-seg.pt"
                                        # works the same way (older YOLOE).
SEGMENT_PROMPTS = ["shoe", "sneaker"]   # yoloe text prompts. We detect single
                                        # shoes (cleanest/most complete -- one
                                        # box per shoe) and then pair them
                                        # geometrically (SEGMENT_PAIR below) into
                                        # one record per tied pair. Prompting for
                                        # "pair of shoes" gave messy overlapping
                                        # boxes, so we don't.
SEGMENT_CONF = 0.10                     # min confidence to keep a segment.
                                        # Tuned 2026-06-12 against the hand-
                                        # counted set (eval_tiling.py): 0.25
                                        # scored 0.655 count-recall; 0.10 with
                                        # yoloe-11m-seg scores 0.880 with ZERO
                                        # false positives on empty tables and
                                        # ~no over-detection -- every error was
                                        # under-detection, so the threshold was
                                        # the bottleneck. Below 0.10 phantom
                                        # boxes appear (0.05 -> 2 FPs on empty
                                        # tables); don't lower further without
                                        # re-running the eval.
SEGMENT_IMGSZ = 1280                    # inference resolution. A whole table of
                                        # many small shoes is the hard case: the
                                        # default 640 downsamples them away. Push
                                        # this toward the photo's real size
                                        # (e.g. 1280/1536) so distant pairs
                                        # survive. Higher = slower but far better
                                        # recall on dense tables. Must be /32.
SEGMENT_DEVICE = "auto"                 # "auto" (reuses YOLO device pick) or
                                        # "cpu"/"mps"/"cuda:0"
# Tiling (SAHI-style): a single wide pass misses most shoes on a crowded table
# because each one is tiny. Tiling slices the photo into overlapping windows,
# detects in each (each upscaled to SEGMENT_IMGSZ, so shoes look big), then
# merges. This is the main recall lever for dense tables.
SEGMENT_TILE = 512                      # px; 0 = whole image (no tiling). When
                                        # >0, slice into TILE x TILE windows.
                                        # Smaller TILE = bigger shoes to the
                                        # model = better recall, but more tiles
                                        # = slower.
SEGMENT_TILE_OVERLAP = 0.25             # fraction adjacent tiles overlap, so a
                                        # shoe on a seam is still whole in a
                                        # neighbor tile. 0.2-0.3 is typical.
SEGMENT_TILE_IOU = 0.4                  # merge two detections from different
                                        # tiles when they overlap by >= this IoU
                                        # (the higher-confidence one is kept).
SEGMENT_TILE_IMGSZ = 640                # inference resolution for the TILES
                                        # (the full pass uses SEGMENT_IMGSZ).
                                        # Must stay near the tile size: running
                                        # a 512px tile at 1280 upscales 2.5x and
                                        # collapses confidence below SEGMENT_CONF
                                        # -- tiles silently return nothing.
SEGMENT_MAX_SIDE = 2560                 # cap the photo's long side (px) before
                                        # tiling. Station shots (1920x1080) pass
                                        # through untouched; oversized photos
                                        # (e.g. 8000x6000 phone shots) are
                                        # downscaled for DETECTION only -- boxes
                                        # map back to original coords, so crops
                                        # still come from the full-res photo.
                                        # Bounds tile count (337 -> ~30 on the
                                        # 8000x6000 case) and stops shoes bigger
                                        # than a tile being cut at many seams.
                                        # 0 = no cap.
SEGMENT_TILE_BATCH = 8                  # tiles per GPU predict() call. One call
                                        # per tile (the old behavior, = 1) wastes
                                        # most of the time on fixed per-call
                                        # overhead; chunks of 8 cut segmentation
                                        # wall-time ~2-4x with identical output.
                                        # Bounded (not all-at-once) so VRAM stays
                                        # predictable on the shared GPU (ollama +
                                        # DINOv2 + engine) and the dash website
                                        # never stalls behind a VRAM spike.
SEGMENT_SEAM_PX = 2                     # a tile detection within this many px of
                                        # a tile edge (that isn't an image edge)
                                        # is a truncated partial: quarantined from
                                        # the main merge so it can't suppress the
                                        # complete box of the same shoe.
SEGMENT_RESCUE_CONTAIN = 0.3            # a quarantined partial is rescued (kept)
                                        # only if its containment vs every kept
                                        # box is below this -- i.e. no clean box
                                        # already covers that region. Lower =
                                        # stricter (fewer sliver crops).
SEGMENT_TILER = "custom"                # which tiler to use when SEGMENT_TILE>0:
                                        # "custom" = the hand-rolled TiledSegmenter
                                        # (greedy IoU+containment NMS); "sahi" =
                                        # the SAHI library's slicing + Greedy
                                        # NMM/NMS merge (obss/sahi, MIT). Same
                                        # base model either way -- swap to A/B the
                                        # tiling logic alone. See ab_tiling.py.
SEGMENT_SAHI_MERGE = "NMS"              # sahi tiler only: "NMS" (suppress, keep
                                        # winner box -- closest to custom) or
                                        # "NMM" (merge overlaps into their union).
SEGMENT_SAHI_METRIC = "IOS"            # sahi tiler only: overlap metric for the
                                        # merge -- "IOS" (intersection-over-smaller,
                                        # good for tile-seam partials) or "IOU".

# ---- SAM2 escalation hybrid (research 2026-06-27, OFF by default) -----------
# On crowded / low-contrast tables YOLOE under-detects and produces sliver crops.
# SAM2-everything + a learned shoe/not-shoe gate beats it there (validated: pairs
# +20%, slivers 5->0, recall 0.93 vs 0.88) but is ~56% slower, so it is run only
# as an ESCALATION: the YOLOE tiler runs every time; when its result looks weak
# we additionally run SAM2+gate and keep whichever found MORE shoes (never worse
# than YOLOE alone). Entirely additive -- set SEGMENT_ESCALATE_SAM2=False (or env
# ENGINE_SEGMENT_ESCALATE=0) and the pipeline is byte-identical to before.
SEGMENT_ESCALATE_SAM2 = False          # master switch for the SAM2 escalation.
SEGMENT_ESCALATE_MODE = "weak"          # when to escalate: "weak" = only when the
                                        # YOLOE pass looks weak (see thresholds
                                        # below); "always" = on every photo (max
                                        # effect, for evaluation -- ignores cost).
SEGMENT_ESCALATE_MAX_SHOES = 28         # "weak": escalate if YOLOE kept <= this
                                        # many shoes (crowded tables it under-counts).
SEGMENT_ESCALATE_MIN_SLIVERS = 1        # "weak": OR escalate if YOLOE produced at
                                        # least this many sliver boxes (seam-cut
                                        # partials -- the exact artifact SAM2 fixes).
SEGMENT_ESCALATE_GATE = "dataset/gate_clf/gate_cnn.pt"   # learned shoe/not-shoe
                                        # gate weights (resnet18). Missing file =>
                                        # escalation self-disables (logs, stays YOLOE).
SEGMENT_ESCALATE_SAM_MODEL = "sam2_b.pt"  # SAM2 weights (Apache-2.0).
SEGMENT_ESCALATE_SAM_MAX = 1536        # downscale SAM2 input long side (px) before
                                        # everything-mode, else it OOMs on huge
                                        # photos; boxes are mapped back to source.
SEGMENT_ESCALATE_AF_LO = 0.004         # SAM mask size filter: min area fraction,
SEGMENT_ESCALATE_AF_HI = 0.12          #   max area fraction,
SEGMENT_ESCALATE_AR_MAX = 4.5          #   and max aspect ratio (drop junk masks).

SEGMENT_CROP_PAD = 0.04                 # pad each crop by this fraction of bbox
                                        # size on every side (a little context
                                        # helps the make/model step). 0 = tight.
SEGMENT_APPLY_MASK = False              # if True, white-out everything outside
                                        # the segment polygon in the saved crop
                                        # (cleaner brand crops); needs masks.
SEGMENT_MIN_AREA_FRAC = 0.0             # drop segments smaller than this frac of
                                        # the photo area (noise). 0 = keep all.
SEGMENT_PAIR = True                     # shoes arrive tied in pairs -> group the
                                        # detected single shoes into pairs, one
                                        # record per pair (the union crop). False
                                        # = keep one record per individual shoe.
SEGMENT_PAIR_MAX_GAP = 1.2              # "geometry" method only: pair two shoes
                                        # when the gap between their centers is
                                        # within this multiple of their average
                                        # size. Lower = stricter (won't bridge the
                                        # gap between pairs); higher = more eager.
SEGMENT_PAIR_METHOD = "hybrid"          # "hybrid" (default) = ADJACENCY first
                                        # (workers place mates touching/stacked)
                                        # with a DINOv2 appearance veto + a
                                        # high-bar visual rescue for separated
                                        # mates. Chosen over "visual" because
                                        # measured cosines rank silhouette over
                                        # identity (strangers 0.87 > mates 0.41).
                                        # "visual" = pair by APPEARANCE (DINOv2/
                                        # CLIP embedding similarity + a soft
                                        # spatial tiebreak) so shoes need NOT be
                                        # tied or placed adjacently -- workers can
                                        # just lay them on the table. "geometry" =
                                        # legacy nearest-neighbour (needs tied/
                                        # adjacent pairs; uses SEGMENT_PAIR_MAX_GAP).
SEGMENT_PAIR_SPATIAL_WEIGHT = 0.15      # "visual" only: how much closeness on the
                                        # table breaks ties between similar-looking
                                        # shoes. 0 = pure appearance.
SEGMENT_PAIR_VETO_MIN = 0.25            # "hybrid" only: an ADJACENT candidate is
                                        # rejected when its crops' cosine is below
                                        # this (clearly different objects). True
                                        # mates in awkward poses measured 0.41-0.49,
                                        # so keep this floor well under that.
SEGMENT_PAIR_RESCUE_MIN = 0.80          # "hybrid" only: NON-adjacent shoes pair
                                        # only when they look near-identical
                                        # (cos >= this). High on purpose: a gray
                                        # Brooks scored 0.87 against a white one,
                                        # so rescue is a last resort, not the rule.
SEGMENT_PAIR_MIN_SIM = 0.65            # "visual" only: accept a pair only if its
                                        # blended score (cosine - spatial*dist) is
                                        # >= this; below -> both shoes become
                                        # singles. True-mate cosines measured on
                                        # real tables cluster 0.69-0.85, while
                                        # 0.5 let different-but-similar runners
                                        # pair across the table. TUNE on real
                                        # photos -- the engine logs each pair's
                                        # cosine + score to stderr.
SEGMENT_PAIR_MAX_DIST_FRAC = 0.18      # "visual"+"hybrid": reject a pair whose two
                                        # shoes' centers are farther apart than this
                                        # fraction of the image diagonal. Stops the
                                        # cross-table mis-match whose UNION crop
                                        # spans the whole table (the "terrible crop"
                                        # bug). 0 disables. Lower = tighter pairs.

# --- Crop background whitening (the old-system look) -------------------------
# Paint everything outside the shoe mask(s) white, so each saved crop shows ONLY
# its one/two shoes on a clean white background -- even if a neighbor sits inside
# the union box, it's masked away. Needs per-shoe masks (SAM2 gives clean ones;
# YOLOE-seg too). Fail-safe: a crop with no mask is saved unchanged.
SEGMENT_WHITEN_CROP = True             # mask-out the crop background.
SEGMENT_WHITEN_DILATE = 9              # grow the mask by this many px first, so a
                                        # slightly-tight mask doesn't shave the
                                        # shoe's edge. 0 = exact mask.
SEGMENT_WHITEN_COLOR = 205             # background grayscale value (0-255). NOT
                                        # pure white: 255 makes white shoes/laces
                                        # dissolve into the background (lost
                                        # silhouette). ~205 mid-gray keeps white
                                        # shoes' edges; raise toward 255 for a
                                        # whiter look, lower for more contrast.
SEGMENT_CROP_MASK_SAM2 = True          # refine each crop's whitening mask with
                                        # SAM2 box-prompt (clean, tight masks)
                                        # instead of the loose YOLOE detection
                                        # mask. Adds ~4-6s/photo (box-prompt, far
                                        # cheaper than SAM2 everything-mode) and
                                        # makes whitened crops clean on EVERY
                                        # table. Only runs when WHITEN_CROP is on.
SEGMENT_WHITEN_TIGHTEN = True          # after whitening, re-crop to the shoe
                                        # mask's bounding box so the shoes fill
                                        # the frame (no big gray margin); also
                                        # drops a neighbor that fell in the loose
                                        # union box. Off = keep the padded bbox.
SEGMENT_WHITEN_TIGHTEN_MARGIN = 10     # px of breathing room around the mask
                                        # when tightening.

# Where whole-table photos are read from and where per-pair crops are written.
TABLE_INPUT_DIR = "sneaker_impact/table_photos"   # full-table photos land here
TABLE_OUTPUT_ROOT = "sneaker_impact/pairs"        # per-pair crops + JSON go here
TABLE_PHOTO_PREFIX = "table"            # incoming table photos are renamed
                                        # table1.jpg, table2.jpg, ... by
                                        # ingest_table.py, so every per-pair
                                        # record's source_photo is logical and
                                        # traceable (table3 -> pair_7, etc.).

# --- Brand recognition (2026 pivot, Phase B) ------------------------------
# Fill each pair's `make` field (the brand). Pluggable like the segmenter, so we
# can start local + free now and later swap in a trained classifier (on the
# supercomputer) or a vision-LLM API for higher accuracy.
#   "clip" -- local zero-shot CLIP. No training, no API key, no cost: it compares
#             the crop against text like "a photo of Nike shoes" and picks the
#             closest brand. A solid baseline; logos are small so it won't be
#             perfect -- the dashboard human-confirm is the safety net.
BRAND_BACKEND = "clip"
BRAND_MODEL = "ViT-B/32"                # CLIP variant (~340MB, auto-downloads
                                        # once). Bigger (ViT-L/14) = better/slower.
BRAND_DEVICE = "auto"                   # "auto" (CUDA->MPS->CPU) or cpu/mps/cuda
BRAND_CLASSES = [                       # brands to choose from -- edit freely
    "Nike", "Jordan", "Adidas", "Yeezy", "New Balance", "Converse", "Vans",
    "Puma", "Reebok", "Asics", "Under Armour", "Saucony", "Brooks", "Hoka",
    "Skechers", "Fila", "Salomon",
]
BRAND_PROMPT = "a photo of {} shoes"    # CLIP text template; {} = brand name
BRAND_MIN_CONF = 0.35                   # if the top brand scores below this,
                                        # label "unknown" instead of guessing.
                                        # Zero-shot CLIP is confident+right on
                                        # iconic brands (NB, Jordan, 3-stripes)
                                        # but coin-flips plain shoes, so we don't
                                        # commit weak guesses as fact -- they go
                                        # to human/Phase-C review. 0 = always guess.

# --- Curated label_data export (Phase B) ----------------------------------
# Copy only the CONFIDENTLY labeled pairs -- both color (Phase A) and make
# (Phase B) above their thresholds, and neither "unknown"/"multi" -- into a
# clean, training-ready folder, named shoes_<color>_<make>_<N>.jpg
# (e.g. shoes_blue_newBalance_1.jpg). This is the high-quality subset to hand to
# the (out-of-scope) training step. Export is idempotent (dedups by source).
LABEL_DATA_DIR = "label_data"
LABEL_MAKE_MIN_CONF = 0.60              # brand confidence required to export
LABEL_COLOR_MIN_CONF = 0.45             # color confidence required to export
                                        # (0.45 so legit two-tone shoes like a
                                        # white+black-stripe Adidas, ~0.49, still
                                        # qualify)

# --- Model recognition (2026 pivot, Phase C) ------------------------------
# Identify the specific MODEL (e.g. "Air Jordan 1") from the crop + the brand
# Phase B found. Pluggable like the others:
#   "ollama"     -- a LOCAL Ollama vision model (free, private, on-device). Sends
#                   the crop + brand and parses {model, confidence}. NOTE: the
#                   VLM's self-reported confidence is NOT calibrated (often a flat
#                   0.95), so treat "unknown" as the real signal.
#   "clip-index" -- match the crop against the reverse-image index for a REAL
#                   similarity + source (needs build_catalog_index.py first).
#   "hybrid"     -- VLM proposes the name, the index verifies it: a non-null
#                   model_confidence then means "index-verified" (with a source),
#                   and a disagreement is flagged for the human. RECOMMENDED once
#                   a good index exists; needs BOTH Ollama running and an index.
MODEL_BACKEND = "ollama"
MODEL_OLLAMA_MODEL = "qwen2.5vl:7b"     # on the supercomputer: qwen3-vl:32b etc.
MODEL_OLLAMA_URL = "http://localhost:11434"   # local Ollama server
MODEL_OLLAMA_TIMEOUT = 180              # seconds per image (first load is slow)
MODEL_MIN_CONF = 0.0                    # below this confidence -> "unknown". VLM
                                        # confidence is uncalibrated, so 0 keeps
                                        # its answer; the real gate is the
                                        # human-confirm step.

# CLIP reverse-image index -- the "second opinion" verifier. Matches a crop
# against a catalog of known sneaker images (built by build_catalog_index.py)
# and returns the nearest model + a REAL similarity score + a source link.
# Catalog merges two sources: a public dataset dropped under CLIP_CATALOG_DIR
# (organized <brand>/<model>/*.jpg) AND our growing label_data. Select it with
# MODEL_BACKEND="clip-index"; rebuild the index whenever the catalog changes.
# --- Reverse-image index embedder ----------------------------------------
# The function that turns an image into a vector for the "clip-index" verifier.
# Both build_catalog_index.py and the query side (model_search.py) read this ONE
# setting (via embedder_utils), so the index is always built AND queried with the
# same embedder. CHANGING THIS REQUIRES REBUILDING THE INDEX (different vector
# size); the query side detects a stale index and tells you to rebuild.
#   "clip"   -- OpenAI CLIP ViT-B/32 (512-d). The original baseline. Fast, but
#               NOT good at fine-grained sneaker retrieval (crosses brands; the
#               scores don't separate right matches from wrong).
#   "dinov2" -- Meta DINOv2 (self-supervised, built for instance / fine-grained
#               retrieval) -- much better at "same model, different photo".
#               Local + free (downloaded once via torch.hub). Recommended.
EMBED_BACKEND = "dinov2"
EMBED_DINOV2_MODEL = "dinov2_vitl14_reg"   # vits14/vitb14/vitl14/vitg14 (+ _reg);
                                           # bigger = better + more VRAM. The
                                           # supercomputer can afford _vitl14/g14.
EMBED_DEVICE = "auto"                  # auto = CUDA -> MPS -> CPU (pick_device)

CLIP_CATALOG_DIR = "sneaker_impact/catalog"        # drop a public dataset here
CLIP_DATASET_DIRS = [                              # flat <brand>_<model>/*.jpg
    # public datasets (brand inferred from the class-folder name); combine many.
    "downloads/popular_sneakers/sneakers-dataset/sneakers-dataset",
]
CLIP_INDEX_PATH = "sneaker_impact/clip_index.npz"  # built index (embeddings)
CLIP_INDEX_MODEL = "ViT-B/32"          # CLIP variant -- only used when
                                       # EMBED_BACKEND="clip" (build+query match).
CLIP_INDEX_MIN_SIM = 0.90              # cosine similarity below this -> "unknown".
                                       # RE-TUNE PER EMBEDDER: this 0.90 was set
                                       # for CLIP (which rates DIFFERENT sneakers
                                       # ~0.8). DINOv2's similarity distribution
                                       # is different, so re-measure on a held-out
                                       # set (correct vs. wrong matches) and pick
                                       # a threshold that actually separates them.
CLIP_INDEX_BRAND_FILTER = True         # only compare against catalog entries of
                                       # the same (Phase B) brand -- faster + more
                                       # accurate. False = search all brands.