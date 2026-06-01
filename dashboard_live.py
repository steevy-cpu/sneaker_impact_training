"""
dashboard_live.py -- background live push from label_live to the dashboard.

When config.DASHBOARD_PUSH_LIVE is on, label_live hands each freshly saved shoe
to a DashboardPusher, which pushes it on a background daemon thread. Fire-and-
forget and fail-safe: the capture loop never blocks or crashes, and if the
dashboard is unreachable the shoe is simply skipped here -- it stays on disk and
dashboard_sync.py can back-fill it later. Pushes are recorded in the SAME ledger
as the sync tool, so the two never double-push the same shoe.

Notes:
  - Don't run dashboard_sync.py while a live-push session is active -- both write
    the ledger file.
  - Undo (U key) deletes the local files. If that happens before the pusher gets
    to the shoe, it's skipped naturally (the file is gone). If it was already
    pushed, the dashboard record remains -- the dashboard has no delete API.
"""
import json
import os
import queue
import threading

from dashboard_client import load_ledger, save_ledger


def _read_meta(jpg_path):
    """Load the .json sidecar next to a saved crop, or None."""
    json_path = (jpg_path[:-4] + ".json") if jpg_path.endswith(".jpg") else jpg_path + ".json"
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:                              # noqa: BLE001 - fail safe
        return None


class DashboardPusher:
    """Pushes saved shoes to the dashboard on a background thread."""

    def __init__(self, client, ledger_path):
        self.client = client
        self.ledger_path = ledger_path
        self.ledger = load_ledger(ledger_path)
        self.queue = queue.Queue()
        self.batch_id = None                       # opened lazily on first push
        self._running = threading.Event()
        self._thread = None

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="dashboard-push")
        self._thread.start()

    def enqueue(self, jpg_path):
        """Queue a saved crop for pushing (non-blocking; safe from main loop)."""
        self.queue.put(jpg_path)

    def stop(self):
        """Stop after draining queued pushes (bounded wait)."""
        self._running.clear()
        self.queue.put(None)                       # sentinel
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # --- worker thread ---------------------------------------------------

    def _run(self):
        while True:
            try:
                jpg = self.queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if jpg is None:                        # sentinel from stop()
                break
            try:
                self._push_one(jpg)
            except Exception as exc:               # never let the thread die
                print(f"[dashboard] live push error: {exc}")

    def _push_one(self, jpg):
        if jpg in self.ledger["shoes"]:
            return                                 # already pushed (run or sync)
        if not os.path.exists(jpg):
            return                                 # deleted (undone) before push
        meta = _read_meta(jpg)
        if meta is None:
            return
        if self.batch_id is None:
            self.batch_id = self.client.open_batch()
            if not self.batch_id:
                # Dashboard down: skip; dashboard_sync.py back-fills it later.
                print("[dashboard] live push: dashboard unreachable, skipping "
                      "(dashboard_sync.py can back-fill later)")
                return
        shoe_id = self.client.push_shoe(jpg, meta, self.batch_id)
        if shoe_id:
            self.ledger["shoes"][jpg] = shoe_id
            save_ledger(self.ledger_path, self.ledger)
