"""
list_cameras.py -- camera discovery utility.

Probes camera indices 0 through 5 and reports which ones actually open and
deliver a frame, so you can pick the right index for your built-in webcam vs.
an external USB-C camera. Then set CAMERA_INDEX in config.py.

    python list_cameras.py
"""
import cv2

from camera_utils import get_camera_backend, release_camera


def probe(index):
    """Try to open `index` and read one frame. Return (works, width, height)."""
    backend = get_camera_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)   # fall back to default backend
    if not cap.isOpened():
        release_camera(cap)
        return False, 0, 0

    ok, frame = cap.read()
    release_camera(cap)
    if ok and frame is not None:
        h, w = frame.shape[:2]
        return True, w, h
    return False, 0, 0


def main():
    print("Probing camera indices 0..5 ...\n")
    working = []
    for i in range(6):
        works, w, h = probe(i)
        if works:
            print(f"  [{i}] WORKS  ({w}x{h})")
            working.append(i)
        else:
            print(f"  [{i}] not available")

    print()
    if not working:
        print("No working cameras found.")
        print("Check the USB-C connection and camera permissions, then retry.")
        return

    print(f"Working camera indices: {working}")
    # Built-in webcams are usually index 0; an external USB-C camera is often
    # the higher index, which is probably what you want.
    suggested = working[-1] if len(working) > 1 else working[0]
    print(f"Suggested CAMERA_INDEX for config.py: {suggested}")
    if len(working) > 1:
        print("(Index 0 is usually the built-in webcam; the higher index is "
              "likely your external USB-C camera.)")


if __name__ == "__main__":
    main()
