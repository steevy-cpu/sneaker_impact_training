"""
list_cameras.py -- camera discovery utility.

Probes camera indices 0 through 5 and reports which ones actually open and
deliver a frame, including each camera's name (on macOS). Use it to pick your
external USB-C camera, then either:
    - set CAMERA_NAME in config.py to part of its name (recommended), or
    - set CAMERA_INDEX in config.py to its index.

    python list_cameras.py
"""
import cv2

from camera_utils import get_camera_backend, list_camera_names, release_camera


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
    names = list_camera_names()   # device names in index order (macOS); else []
    print("Probing camera indices 0..5 ...\n")
    working = []
    for i in range(6):
        works, w, h = probe(i)
        name = names[i] if i < len(names) else ""
        name_part = f"  \"{name}\"" if name else ""
        if works:
            print(f"  [{i}] WORKS  ({w}x{h}){name_part}")
            working.append((i, name))
        else:
            print(f"  [{i}] not available{name_part}")

    print()
    if not working:
        print("No working cameras found.")
        print("Check the USB-C connection and camera permissions, then retry.")
        return

    # Prefer a camera that is clearly NOT the built-in webcam.
    builtin_hints = ("facetime", "built-in", "builtin")
    external = [(i, n) for i, n in working
                if n and not any(h in n.lower() for h in builtin_hints)]

    print(f"Working cameras: {[i for i, _ in working]}")
    if external:
        idx, name = external[0]
        # Suggest a short, distinctive substring of the name for CAMERA_NAME.
        hint = name.split()[0] if name else ""
        print(f"\nLooks like your external camera is index {idx}: \"{name}\"")
        print(f"Recommended (robust): set CAMERA_NAME = \"{hint}\" in config.py")
        print(f"Or set CAMERA_INDEX = {idx}")
    else:
        idx = working[-1][0] if len(working) > 1 else working[0][0]
        print(f"\nSuggested CAMERA_INDEX for config.py: {idx}")
        print("(Plug in your USB-C camera and re-run to select it by name.)")


if __name__ == "__main__":
    main()
