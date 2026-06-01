"""
log_utils.py -- mirror console output to a timestamped log file.

The app uses print() throughout (beginner-readable). start_file_logging() tees
stdout + stderr to a file so an unattended capture session keeps a record
without changing any print() call. Best-effort: a file-write error never
disrupts the console or the app.
"""
import os
import sys
from datetime import datetime


class _Tee:
    """Write-through wrapper: everything written to the console also goes to a
    file. File errors are swallowed so logging can never take down the app."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass


def start_file_logging(log_dir):
    """Tee stdout + stderr into `log_dir/label_live_<timestamp>.log`.

    Returns the log path, or None if logging couldn't be set up (never raises).
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        path = os.path.join(log_dir, f"label_live_{stamp}.log")
        fh = open(path, "a", buffering=1)
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)
        print(f"[log] mirroring console to {path}")
        return path
    except Exception as exc:                       # noqa: BLE001 - never crash
        print(f"[log] file logging disabled: {exc}")
        return None
