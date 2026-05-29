"""
bayer_test.py -- find the correct Bayer pattern for the Photon Focus camera.

Captures one frame and saves it decoded with all 4 Bayer patterns as:
  /tmp/bayer_1_BayerGB8.png
  /tmp/bayer_2_BayerRG8.png
  /tmp/bayer_3_BayerGR8.png
  /tmp/bayer_4_BayerBG8.png

Open all four in the file manager and pick whichever looks like natural colors.
Then set GIGE_BAYER_OVERRIDE to the matching name in config.py.
"""
import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import numpy as np
import cv2
import config
import time

PATTERNS = [
    ("BayerGB8", cv2.COLOR_BayerGB2BGR),
    ("BayerRG8", cv2.COLOR_BayerRG2BGR),
    ("BayerGR8", cv2.COLOR_BayerGR2BGR),
    ("BayerBG8", cv2.COLOR_BayerBG2BGR),
]

Aravis.update_device_list()
camera = Aravis.Camera.new(None)
camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
if camera.is_gv_device() and config.GIGE_PACKET_SIZE > 0:
    camera.gv_set_packet_size(config.GIGE_PACKET_SIZE)

# Reset ROI to full sensor
sensor_w = camera.get_integer("WidthMax")
sensor_h = camera.get_integer("HeightMax")
camera.set_region(0, 0, sensor_w, sensor_h)
print(f"ROI: {sensor_w}x{sensor_h}")

camera.set_frame_rate(5.0)
payload = camera.get_payload()
print(f"Payload: {payload} bytes, FPS: {camera.get_frame_rate():.1f}")

stream = camera.create_stream(None, None)
for _ in range(30):
    stream.push_buffer(Aravis.Buffer.new_allocate(payload))

camera.start_acquisition()
print("Waiting for a clean frame...")

# Wait for a good frame
raw = None
for attempt in range(10):
    buf = stream.timeout_pop_buffer(2_000_000)
    if buf is None:
        print(f"  attempt {attempt+1}: timeout")
        continue
    status = buf.get_status()
    data = buf.get_data()
    bw = buf.get_image_width()
    bh = buf.get_image_height()
    expected = bw * bh
    print(f"  attempt {attempt+1}: status={status.value_nick} size={len(data)} {bw}x{bh}")
    if status == Aravis.BufferStatus.SUCCESS and len(data) > 0:
        # Use actual received size — camera may ignore ROI changes until power-cycled.
        actual_w = len(data) // bh if bh > 0 else bw
        raw = np.frombuffer(data, dtype=np.uint8)[:actual_w * bh].copy()
        raw = raw.reshape(bh, actual_w)
        print(f"  actual frame size: {actual_w}x{bh}")
        stream.push_buffer(buf)
        break
    stream.push_buffer(buf)
    time.sleep(0.2)

camera.stop_acquisition()

if raw is None:
    print("\nERROR: could not capture a clean frame.")
    print("Check 12V power, Ethernet cable, and that eth0 is on 192.168.55.x subnet.")
else:
    print(f"\nGot frame {raw.shape}, min={raw.min()} max={raw.max()} mean={raw.mean():.1f}")
    # Also save the raw Bayer plane (grayscale) for reference
    cv2.imwrite("/tmp/bayer_0_RAW.png", raw)
    print("Saved /tmp/bayer_0_RAW.png  (raw Bayer, no decode)")
    for i, (name, code) in enumerate(PATTERNS, 1):
        bgr = cv2.cvtColor(raw, code)
        path = f"/tmp/bayer_{i}_{name}.png"
        cv2.imwrite(path, bgr)
        print(f"Saved {path}")
    print("\nOpen the /tmp/bayer_*.png files in the file manager.")
    print("Pick the one that looks like natural colors.")
    print("Then set in config.py:  GIGE_BAYER_OVERRIDE = \"<name>\"")
