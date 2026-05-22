"""
Shoe dataset capture tool.

Opens the webcam and uses a YOLOv8 model pretrained on Open Images V7 to
detect *shoes specifically* (Footwear / Boot / Sandal / High heels classes).
When a shoe is in frame you can press a label key to crop it out and save it
into a labelled folder.

Labels:
    1 -> A   (GOOD TOP shoe)
    2 -> B   (BAD  TOP shoe)
    3 -> A2  (GOOD BOTTOM shoe)
    4 -> B2  (BAD  BOTTOM shoe)

Other keys:
    q / ESC  -> quit
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# key (as char) -> (folder name, human description)
LABELS = {
    "1": ("A", "GOOD TOP"),
    "2": ("B", "BAD TOP"),
    "3": ("A2", "GOOD BOTTOM"),
    "4": ("B2", "BAD BOTTOM"),
}

# Open Images V7 class ids that count as a shoe.
SHOE_CLASS_IDS = {203, 56, 432, 249}  # Footwear, Boot, Sandal, High heels

# Bottom-of-shoe labels: the detector can't recognize a sole, so these use
# GrabCut on a center guide box instead of the shoe detector.
BOTTOM_KEYS = {"3", "4"}


def make_dirs(root: Path) -> None:
    for folder, _ in LABELS.values():
        (root / folder).mkdir(parents=True, exist_ok=True)


def pick_shoe(result):
    """Return (box, conf, class_name) of the largest detected shoe, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_area = 0.0
    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        if cls not in SHOE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        area = float((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best = (
                (x1, y1, x2, y2),
                float(boxes.conf[i].item()),
                result.names[cls],
            )
    return best


def crop(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def center_rect(shape, frac=0.6):
    """A centered (x, y, w, h) box covering `frac` of the frame."""
    h, w = shape[:2]
    rw, rh = int(w * frac), int(h * frac)
    x, y = (w - rw) // 2, (h - rh) // 2
    return x, y, rw, rh


def grabcut_isolate(frame, rect, iters=5):
    """Cut the foreground inside `rect` from its background using GrabCut.

    Returns a tightly-cropped BGRA image (background transparent), or None if
    no foreground was found.
    """
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, mask, tuple(rect), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    crop_img = frame[y1:y2 + 1, x1:x2 + 1]
    crop_mask = fg[y1:y2 + 1, x1:x2 + 1]
    bgra = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = crop_mask
    return bgra


def draw_overlay(frame, box, conf, name):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_legend(frame, detected):
    lines = [
        "1=A GOOD TOP   2=B BAD TOP   (needs green box)",
        "3=A2 GOOD BOT  4=B2 BAD BOT  (aim sole in blue box)",
        "q=quit",
    ]
    status = "SHOE DETECTED - press 1/2" if detected else "no shoe (top) - bottoms 3/4 still work"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 52 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Capture and label shoe images for the dataset.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--model", default="yolov8m-oiv7.pt",
                    help="YOLOv8 Open Images V7 model with a Footwear class (auto-downloads)")
    ap.add_argument("--out", default="dataset", help="output dataset root folder")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=480,
                    help="inference resolution; lower = faster/lighter (try 320)")
    ap.add_argument("--every", type=int, default=2,
                    help="run detection every Nth frame (reuse result between); higher = lighter")
    ap.add_argument("--debug", action="store_true",
                    help="draw ALL detections (any class) to see what the model perceives")
    args = ap.parse_args()

    root = Path(args.out)
    make_dirs(root)

    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Webcam open. Show a shoe and press 1/2/3/4 to save, q to quit.")
    counts = {folder: 0 for folder, _ in LABELS.values()}

    frame_idx = 0
    result = None
    shoe = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            # Only run the detector every Nth frame to lighten the load; reuse
            # the previous detection on the frames in between.
            if frame_idx % max(args.every, 1) == 0:
                result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                shoe = pick_shoe(result)
            frame_idx += 1

            view = frame.copy()
            if args.debug and result is not None:
                for i in range(len(result.boxes)):
                    b = result.boxes.xyxy[i].cpu().numpy()
                    c = float(result.boxes.conf[i].item())
                    nm = result.names[int(result.boxes.cls[i].item())]
                    x1, y1, x2, y2 = [int(v) for v in b]
                    cv2.rectangle(view, (x1, y1), (x2, y2), (255, 180, 0), 1)
                    cv2.putText(view, f"{nm} {c:.2f}", (x1, max(y1 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)
            # blue guide box for bottom (sole) captures
            gx, gy, gw, gh = center_rect(frame.shape)
            cv2.rectangle(view, (gx, gy), (gx + gw, gy + gh), (255, 120, 0), 2)
            if shoe is not None:
                box, conf, name = shoe
                view = draw_overlay(view, box, conf, name)
            view = draw_legend(view, shoe is not None)

            cv2.imshow("Shoe capture", view)
            key = cv2.waitKey(1) & 0xFF
            char = chr(key) if key != 255 else ""

            if key in (ord("q"), 27):
                break

            if char in LABELS:
                folder, desc = LABELS[char]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{int(time.time()*1000)%1000:03d}"

                if char in BOTTOM_KEYS:
                    # sole: detector can't see it, so isolate via GrabCut on the guide box
                    print("  isolating sole (GrabCut), hold still...")
                    img = grabcut_isolate(frame, center_rect(frame.shape))
                    if img is None:
                        print("  GrabCut found no foreground - nothing saved")
                        continue
                    path = root / folder / f"{folder}_{ts}.png"
                    cv2.imwrite(str(path), img)
                    src = "grabcut"
                else:
                    # top: require a shoe detection, save bbox crop
                    if shoe is None:
                        print("  no shoe detected - nothing saved")
                        continue
                    box, conf, name = shoe
                    img = crop(frame, box)
                    if img is None:
                        print("  invalid crop - nothing saved")
                        continue
                    path = root / folder / f"{folder}_{ts}.jpg"
                    cv2.imwrite(str(path), img)
                    src = f"{name} {conf:.2f}"

                counts[folder] += 1
                print(f"  saved [{folder} = {desc}] ({src}) -> {path} "
                      f"(total in {folder}: {counts[folder]})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done. Saved per folder:", counts)


if __name__ == "__main__":
    main()
