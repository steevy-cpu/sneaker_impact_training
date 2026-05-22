"""
tracking_utils.py -- lightweight shoe tracking (PLACEHOLDER).

FUTURE PURPOSE
--------------
Give each shoe a stable temporary identity while it stays in frame, so the
operator's label sticks to the right shoe. Planned approach is simple and
dependency-free (centroid tracking / IoU matching) -- no heavy tracking
frameworks initially.

Each tracked shoe will hold:
    - bbox
    - classification state (Reuse / Recycle)
    - time first seen
    - time last seen
    - saved status

When a shoe disappears for config.TRACK_EXPIRATION_FRAMES frames, its save
operation is finalized.

Implementation arrives in Phase 3. Nothing is implemented yet.
"""
