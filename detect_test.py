r"""
Detection tester — see what the shoe model perceives, live.

Use this to check whether the detector can recognize the BOTTOM (sole) of a
shoe. Run it, hold the sole up to the camera, and watch the boxes/labels.

Live mode (default):
    python detect_test.py
    - Opens a window, runs the model every frame at a low threshold.
    - Draws EVERY detection (any class) with its name + confidence.
    - Shoe classes (Footwear/Boot/Sandal/High heels) are highlighted in green
      and announced in the terminal; everything else is drawn in orange.
    - q / ESC to quit.

One-shot mode:
    python detect_test.py --shot
    - Grabs a single frame, prints every detection, saves debug_frame.jpg.
"""

import argparse
import cv2
from ultralytics import YOLO

import config
from camera_utils import open_camera, release_camera

SHOE_CLASS_IDS = {203, 56, 432, 249}  # Footwear, Boot, Sandal, High heels


def draw_detections(view, result):
    """Draw all detections; return list of (conf, name) for shoe-class hits."""
    shoe_hits = []
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        name = result.names[cls]
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
        is_shoe = cls in SHOE_CLASS_IDS
        color = (0, 255, 0) if is_shoe else (255, 180, 0)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2 if is_shoe else 1)
        cv2.putText(view, f"{name} {conf:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if is_shoe:
            shoe_hits.append((conf, name))
    return shoe_hits


def run_live(model, cam, conf, imgsz, every):
    cap = open_camera(cam)
    if cap is None:
        raise SystemExit(f"Could not open camera index {cam}. Try --camera 1.")
    print("Live. Hold the SOLE up to the camera and watch. q to quit.")
    last = ""
    frame_idx = 0
    result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break
            if frame_idx % max(every, 1) == 0:
                result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            frame_idx += 1
            if result is None:
                continue
            view = frame.copy()
            shoe_hits = draw_detections(view, result)

            if shoe_hits:
                best = max(shoe_hits)
                msg = f"SHOE-CLASS HIT: {best[1]} {best[0]:.2f}"
                cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                msg = "no shoe-class detection"
                cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if msg != last:
                print("  " + msg)
                last = msg

            cv2.imshow("Detection test", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        release_camera(cap)
        cv2.destroyAllWindows()


def run_shot(model, cam, conf, warmup):
    cap = open_camera(cam)
    if cap is None:
        raise SystemExit(f"Could not open camera index {cam}. Try --camera 1.")
    frame, ok = None, False
    for _ in range(warmup):
        ok, frame = cap.read()
    release_camera(cap)
    if frame is None or not ok:
        raise SystemExit("Camera opened but returned no frame. Try a different --camera index.")

    result = model.predict(frame, conf=conf, verbose=False)[0]
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    print(f"\nTotal detections at conf>={conf}: {n}")
    dets = [(float(boxes.conf[i].item()), result.names[int(boxes.cls[i].item())])
            for i in range(n)]
    for c, name in sorted(dets, reverse=True):
        tag = "  <-- SHOE CLASS" if name in (result.names[i] for i in SHOE_CLASS_IDS) else ""
        print(f"  {c:.2f}  {name}{tag}")
    if n == 0:
        print("  (nothing detected at all — likely a camera/lighting issue)")
    annotated = result.plot()
    cv2.imwrite("debug_frame.jpg", annotated)
    print("\nSaved annotated image to debug_frame.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--model", default="yolov8m-oiv7.pt")
    ap.add_argument("--conf", type=float, default=0.05, help="low so we see everything")
    ap.add_argument("--imgsz", type=int, default=480, help="inference resolution; lower = lighter")
    ap.add_argument("--every", type=int, default=2, help="run detection every Nth frame (live)")
    ap.add_argument("--shot", action="store_true", help="one-shot snapshot instead of live")
    ap.add_argument("--warmup", type=int, default=15, help="frames to skip in --shot mode")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)
    if args.shot:
        run_shot(model, args.camera, args.conf, args.warmup)
    else:
        run_live(model, args.camera, args.conf, args.imgsz, args.every)


if __name__ == "__main__":
    main()
