"""
gige_camera.py -- capture frames from a GigE Vision camera (Photon Focus) via Aravis.

WHY THIS EXISTS
---------------
The Photon Focus camera is a GigE Vision industrial camera. OpenCV's
cv2.VideoCapture CANNOT open it -- that path is for USB/webcam devices only.
The open-source Aravis library speaks GigE Vision; this module wraps it in a
tiny object (AravisCapture) that looks just like a cv2.VideoCapture: it has
.read() and .release(). So the rest of the app -- label_live.py, capture.py,
detect_test.py -- doesn't need to change. They call open_camera() (in
camera_utils.py); when config.USE_GIGE_CAMERA is True that returns one of these
instead of a cv2.VideoCapture.

REQUIREMENTS (system packages, NOT pip-installable)
---------------------------------------------------
Linux / Raspberry Pi 5:
    sudo apt install aravis-tools gir1.2-aravis-0.8 python3-gi
macOS (harder; the Pi is the preferred target):
    brew install aravis    + PyGObject so `import gi` works

FAIL-SAFE
---------
If Aravis or the camera isn't available, open_gige_camera() prints a clear
message and returns None -- exactly like the USB path does on failure. Importing
this module never crashes even when `gi` is missing.
"""
import numpy as np
import cv2

import config

# Aravis is reached through PyGObject (`gi`). It's a system package, so it may
# not be installed (e.g. on the Mac dev box). Guard the import so this module
# still loads; open_gige_camera() reports the problem cleanly if it's missing.
try:
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    _ARAVIS_AVAILABLE = True
    _ARAVIS_IMPORT_ERROR = None
except Exception as exc:                       # ImportError, ValueError, etc.
    _ARAVIS_AVAILABLE = False
    _ARAVIS_IMPORT_ERROR = exc

# How many stream buffers to allocate. The camera fills these in a ring while
# we process; ~10 is plenty for a single-consumer loop.
_N_BUFFERS = 10

# Map a GigE/GenICam Bayer pixel-format name to the OpenCV debayer code that
# turns it into BGR (OpenCV's standard channel order).
#
# IMPORTANT: OpenCV's Bayer naming is famously "off by one" from GenICam's. The
# mapping below is the usual best guess, but if RED and BLUE look SWAPPED in the
# live view, change the camera's format's entry here (e.g. BayerGB8 ->
# COLOR_BayerBG2BGR). You only confirm this by looking at a real frame.
_BAYER_TO_BGR = {
    "BayerGB8": cv2.COLOR_BayerGB2BGR,
    "BayerRG8": cv2.COLOR_BayerRG2BGR,
    "BayerGR8": cv2.COLOR_BayerGR2BGR,
    "BayerBG8": cv2.COLOR_BayerBG2BGR,
}

# config.GIGE_BAYER_OVERRIDE lets you swap the Bayer decode without touching
# this file -- useful when the camera reports the wrong pattern.
if config.GIGE_BAYER_OVERRIDE and config.GIGE_BAYER_OVERRIDE in _BAYER_TO_BGR:
    _BAYER_TO_BGR = {k: _BAYER_TO_BGR[config.GIGE_BAYER_OVERRIDE]
                     for k in _BAYER_TO_BGR}
    print(f"[gige] Bayer override active: all formats -> {config.GIGE_BAYER_OVERRIDE}")


class AravisCapture:
    """Minimal cv2.VideoCapture look-alike backed by an Aravis GigE camera.

    Implements only the slice of the VideoCapture API the app actually uses:
        read()     -> (ok: bool, frame: np.ndarray|None)   frame is BGR uint8
        release()  -> stop streaming and free the camera
        isOpened() -> bool
        get(prop)  -> width/height for cv2.CAP_PROP_FRAME_WIDTH / _HEIGHT
    """

    def __init__(self, camera, stream, width, height, pixel_format):
        self._camera = camera
        self._stream = stream
        self._width = width
        self._height = height
        self._pixel_format = pixel_format        # e.g. "BayerGB8", "Mono8"
        self._opened = True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def read(self):
        """Pop the next frame and return (ok, BGR_frame). Never raises."""
        if not self._opened:
            return False, None
        try:
            # Drain any stale buffers that piled up during slow processing
            # (e.g. YOLO inference). Keep only the freshest one so the live
            # view stays current instead of replaying seconds-old frames.
            buffer = self._stream.try_pop_buffer()
            while buffer is not None:
                next_buf = self._stream.try_pop_buffer()
                if next_buf is None:
                    break
                self._stream.push_buffer(buffer)
                buffer = next_buf

            # Nothing was immediately available -- wait up to 1 second.
            if buffer is None:
                buffer = self._stream.timeout_pop_buffer(1_000_000)
            if buffer is None:
                return False, None
            try:
                if buffer.get_status() != Aravis.BufferStatus.SUCCESS:
                    # Bad frame (e.g. dropped packet during buffer starvation).
                    # Return False but don't crash -- caller can retry next tick.
                    return False, None
                frame = self._to_bgr(buffer)
                return (frame is not None), frame
            finally:
                # ALWAYS hand the buffer back so the camera can refill it,
                # even if conversion failed -- otherwise streaming starves.
                self._stream.push_buffer(buffer)
        except Exception as exc:                 # keep the app alive on any error
            print(f"[gige] read error: {exc}")
            return False, None

    def _to_bgr(self, buffer):
        """Convert one Aravis buffer into a BGR uint8 image, or None."""
        w = buffer.get_image_width() or self._width
        h = buffer.get_image_height() or self._height
        raw = np.frombuffer(buffer.get_data(), dtype=np.uint8)
        fmt = self._pixel_format

        # Use actual received bytes to determine width — the camera may ignore
        # ROI changes and send a different width than what the API reports.
        if h > 0 and raw.size >= h:
            w = raw.size // h

        # Bayer (single raw plane that we debayer into color).
        if fmt in _BAYER_TO_BGR:
            need = w * h
            if raw.size < need:
                return None
            plane = raw[:need].reshape(h, w)
            return cv2.cvtColor(plane, _BAYER_TO_BGR[fmt])

        # Plain grayscale -> 3-channel BGR so downstream code is happy.
        if fmt == "Mono8":
            need = w * h
            if raw.size < need:
                return None
            plane = raw[:need].reshape(h, w)
            return cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)

        # Already-color formats.
        if fmt in ("RGB8", "RGB8Packed"):
            need = w * h * 3
            if raw.size < need:
                return None
            return cv2.cvtColor(raw[:need].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt in ("BGR8", "BGR8Packed"):
            need = w * h * 3
            if raw.size < need:
                return None
            return raw[:need].reshape(h, w, 3).copy()

        # Unknown format: don't guess. Caller treats None frame as a read miss.
        print(f"[gige] unsupported pixel format '{fmt}'. Set the camera to "
              "BayerGB8 or Mono8, or add a mapping in gige_camera.py.")
        return None

    def release(self):
        """Stop acquisition and free the camera (safe to call more than once)."""
        if not self._opened:
            return
        self._opened = False
        try:
            self._camera.stop_acquisition()
        except Exception:
            pass
        # Drop references so Aravis frees the stream/camera.
        self._stream = None
        self._camera = None


def open_gige_camera():
    """Open the configured GigE camera and start streaming.

    Returns an AravisCapture (cv2.VideoCapture-like) on success, or None on
    failure (after printing a clear, actionable message). Selection follows
    config.GIGE_CAMERA_NAME (empty = first camera Aravis finds).
    """
    if not _ARAVIS_AVAILABLE:
        print("[gige] ERROR: Aravis Python bindings (gi) are not available.")
        print(f"[gige] ({_ARAVIS_IMPORT_ERROR})")
        print("[gige] Install: sudo apt install aravis-tools gir1.2-aravis-0.8 python3-gi")
        return None

    # Refresh the device list before asking how many cameras are present.
    Aravis.update_device_list()
    if Aravis.get_n_devices() == 0:
        print("[gige] ERROR: no GigE camera found.")
        print("[gige] Check 12V power, the Ethernet link, and that the camera")
        print("[gige] and this machine share a subnet. Try: arv-tool-0.8")
        return None

    name = config.GIGE_CAMERA_NAME or None       # None -> first camera found
    try:
        camera = Aravis.Camera.new(name)
    except Exception as exc:
        print(f"[gige] ERROR: could not open camera (name={name!r}): {exc}")
        return None

    try:
        camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)

        # GVSP stream packet size matters: USB-Ethernet adapters can't do jumbo
        # frames, and a too-large packet silently yields ZERO frames. Only GigE
        # ("gv") devices have this knob.
        if camera.is_gv_device() and config.GIGE_PACKET_SIZE > 0:
            camera.gv_set_packet_size(config.GIGE_PACKET_SIZE)

        # Reset ROI to full sensor so a previously-set partial region doesn't
        # silently crop the image or corrupt the frame geometry.
        try:
            sensor_w = camera.get_integer("WidthMax")
            sensor_h = camera.get_integer("HeightMax")
            camera.set_region(0, 0, sensor_w, sensor_h)
            print(f"[gige] ROI reset to full sensor: {sensor_w}x{sensor_h}.")
        except Exception as exc:
            print(f"[gige] Warning: could not reset ROI: {exc}")

        if config.GIGE_FPS > 0:
            try:
                camera.set_frame_rate(config.GIGE_FPS)
                print(f"[gige] Frame rate capped at {config.GIGE_FPS} fps.")
            except Exception as exc:
                print(f"[gige] Warning: could not set frame rate: {exc}")

        try:
            camera.set_exposure_time(config.GIGE_EXPOSURE_US)
            camera.set_gain(config.GIGE_GAIN)
            print(f"[gige] Exposure: {config.GIGE_EXPOSURE_US} us, Gain: {config.GIGE_GAIN}")
        except Exception as exc:
            print(f"[gige] Warning: could not set exposure/gain: {exc}")

        _x, _y, width, height = camera.get_region()
        pixel_format = camera.get_pixel_format_as_string()
        payload = camera.get_payload()

        stream = camera.create_stream(None, None)
        if stream is None:
            print("[gige] ERROR: could not create the video stream.")
            return None
        for _ in range(_N_BUFFERS):
            stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        camera.start_acquisition()
    except Exception as exc:
        print(f"[gige] ERROR: failed to start streaming: {exc}")
        try:
            camera.stop_acquisition()
        except Exception:
            pass
        return None

    print(f"[gige] OK: streaming from '{camera.get_model_name()}' "
          f"({width}x{height}, {pixel_format}).")
    if pixel_format not in _BAYER_TO_BGR and pixel_format not in (
            "Mono8", "RGB8", "RGB8Packed", "BGR8", "BGR8Packed"):
        print(f"[gige] WARNING: pixel format '{pixel_format}' isn't handled; "
              "frames will come back empty until a mapping is added.")
    return AravisCapture(camera, stream, width, height, pixel_format)
