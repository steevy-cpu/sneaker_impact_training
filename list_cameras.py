"""
list_cameras.py -- camera discovery utility.

Probes camera indices 0 through 5 and reports which ones actually open and
deliver a frame. For each working index it also saves a preview JPG to /tmp/
so you can VISUALLY identify which index is your external camera, then set
CAMERA_INDEX in config.py.

Why preview JPGs? On macOS, the camera names returned by `system_profiler`
don't always match OpenCV's AVFoundation index order -- so picking by name
can grab the wrong camera. Looking at the actual frame is the only reliable
way to confirm which index is which.

    python list_cameras.py
"""
import cv2
import time

from camera_utils import get_camera_backend, list_camera_names, release_camera


def probe(index, save_path=None):
    """Try to open `index`, read a frame, optionally save it.

    Returns (works, width, height). Uses several reads to give the camera
    time to warm up -- some USB cameras return an empty first frame.
    """
    backend = get_camera_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)   # fall back to default backend
    if not cap.isOpened():
        release_camera(cap)
        return False, 0, 0

    ok, frame = False, None
    for _ in range(10):                 # warm-up reads
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    release_camera(cap)
    if ok and frame is not None:
        if save_path:
            cv2.imwrite(save_path, frame)
        h, w = frame.shape[:2]
        return True, w, h
    return False, 0, 0


def main():
    names = list_camera_names()   # device names from system_profiler (macOS)
    print("Probing camera indices 0..5 ...\n")
    working = []
    for i in range(6):
        preview = f"/tmp/cam_{i}.jpg"
        works, w, h = probe(i, save_path=preview)
        # Show the system_profiler name only as a *hint* -- its order may not
        # match OpenCV's index order on macOS, so don't trust it as truth.
        hint = names[i] if i < len(names) else ""
        hint_part = f"  (system_profiler hint: \"{hint}\")" if hint else ""
        if works:
            print(f"  [{i}] WORKS  ({w}x{h})  preview: {preview}{hint_part}")
            working.append(i)
        else:
            print(f"  [{i}] not available{hint_part}")

    print()
    if not working:
        print("No working cameras found.")
        print("Check the USB connection and camera permissions, then retry.")
        return

    print(f"Working cameras: {working}")
    print()
    print("VISUAL CHECK -- open the preview images and pick the one showing")
    print("your external camera's view, then set CAMERA_INDEX in config.py:")
    print(f"    open {'  '.join(f'/tmp/cam_{i}.jpg' for i in working)}")


if __name__ == "__main__":
    main()
