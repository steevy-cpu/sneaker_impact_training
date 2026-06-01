"""
dashboard_client.py -- push collected shoes to the Sneaker Impact Dashboard.

Shared by dashboard_sync.py (backfill) and, later, label_live's live push. Talks
to the dashboard's REST API with the Python stdlib only (urllib) -- no extra
dependency. Everything is fail-safe: a network/disk error logs and returns None
so it can never crash the caller.

Mapping (our saved metadata -> the dashboard's ShoeCreate):
  classification (Reuse/Recycle) -> final_decision AND ai_prediction (mirrored),
                                    review_status = COMPLETED
  detected_color                 -> shoe_color
  model_used (+ " (human-labeled)") -> model_version
  the crop jpg                   -> copied into the dashboard's images/, set img_top
  yolo/color conf, sharpness...  -> notes (traceability)
  ai_confidence                  -> null (avoids false "low confidence" alerts)

The dashboard must run in APP_MODE=actual on this machine, with DASHBOARD_IMAGES_DIR
pointing at its images/ folder.
"""
import json
import os
import shutil
import urllib.error
import urllib.request

_VALID_LABELS = {"REUSE", "RECYCLE"}


def _notes_from_meta(meta):
    """Pack our extra metadata into a short human-readable notes string."""
    parts = []
    yc = meta.get("yolo_confidence")
    if isinstance(yc, (int, float)):
        parts.append(f"yolo_conf={yc:.2f}")
    cc = meta.get("color_confidence")
    if isinstance(cc, (int, float)):
        parts.append(f"color_conf={cc:.2f}")
    sh = meta.get("sharpness")
    if isinstance(sh, (int, float)):
        parts.append(f"sharpness={sh:.0f}")
    if meta.get("tracking_id") is not None:
        parts.append(f"track={meta['tracking_id']}")
    return "  ".join(parts) or None


class DashboardClient:
    """Thin REST client for the Sneaker Impact Dashboard."""

    def __init__(self, base_url, images_dir, operator_id, timeout=10):
        self.base_url = base_url.rstrip("/")
        self.images_dir = images_dir
        self.operator_id = operator_id
        self.timeout = timeout

    # --- HTTP helpers ----------------------------------------------------

    def _post(self, path, payload):
        """POST JSON and return the parsed response dict, or None on error."""
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[dashboard] POST {path} failed: HTTP {exc.code} {body[:200]}")
        except Exception as exc:                   # noqa: BLE001 - fail safe
            print(f"[dashboard] POST {path} error: {exc}")
        return None

    def check(self):
        """Return the dashboard's /api/health dict, or None if unreachable."""
        url = self.base_url + "/api/health"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                   # noqa: BLE001 - fail safe
            print(f"[dashboard] not reachable at {self.base_url}: {exc}")
            return None

    # --- API actions -----------------------------------------------------

    def open_batch(self):
        """Open a new inspection batch; return its id, or None on failure."""
        resp = self._post("/api/batches", {"operator_id": self.operator_id})
        return resp.get("id") if resp else None

    def copy_image(self, jpg_path, batch_id):
        """Copy a crop into the dashboard's images/<batch_id>/ folder.

        Returns the URL the dashboard will serve it at (/images/...), or None.
        """
        try:
            filename = os.path.basename(jpg_path)
            dest_dir = os.path.join(self.images_dir, batch_id)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(jpg_path, os.path.join(dest_dir, filename))
            return f"/images/{batch_id}/{filename}"
        except Exception as exc:                   # noqa: BLE001 - fail safe
            print(f"[dashboard] could not copy {jpg_path}: {exc}")
            return None

    def build_payload(self, meta, label, img_url, batch_id):
        """Build the ShoeCreate body for POST /api/shoes."""
        model = meta.get("model_used") or "unknown"
        return {
            "batch_id":          batch_id,
            "operator_id":       self.operator_id,
            "img_top":           img_url,          # single-camera: only the top slot
            "validation_status": "VALID",
            # No Reuse/Recycle classifier yet -- the human label is the decision,
            # mirrored into ai_prediction so the dashboard's charts populate.
            "ai_prediction":     label,
            "ai_confidence":     None,
            "model_version":     f"{model} (human-labeled)",
            "final_decision":    label,
            "review_status":     "COMPLETED",
            "shoe_color":        meta.get("detected_color"),
            "notes":             _notes_from_meta(meta),
        }

    def push_shoe(self, jpg_path, meta, batch_id):
        """Copy the crop and create one shoe record. Returns the new shoe id,
        or None if the shoe was skipped or the push failed."""
        label = (meta.get("classification") or "").upper()
        if label not in _VALID_LABELS:
            print(f"[dashboard] skip {os.path.basename(jpg_path)}: "
                  f"unexpected label {label!r}")
            return None
        img_url = self.copy_image(jpg_path, batch_id)
        if img_url is None:
            return None
        resp = self._post("/api/shoes",
                          self.build_payload(meta, label, img_url, batch_id))
        return resp.get("id") if resp else None


# --- Sync ledger (shared by dashboard_sync.py and dashboard_live.py) --------
# Maps our crop path -> dashboard shoe id (and source folder -> batch id) so
# neither the backfill tool nor live push ever double-pushes a shoe.
DEFAULT_LEDGER = "dashboard_synced.json"


def load_ledger(path):
    """Read the sync ledger, or return a fresh empty one."""
    try:
        with open(path) as f:
            data = json.load(f)
        data.setdefault("folders", {})
        data.setdefault("shoes", {})
        return data
    except (FileNotFoundError, ValueError):
        return {"folders": {}, "shoes": {}}


def save_ledger(path, ledger):
    """Write the sync ledger (best-effort; never raises)."""
    try:
        with open(path, "w") as f:
            json.dump(ledger, f, indent=2)
    except Exception as exc:                       # noqa: BLE001 - non-fatal
        print(f"[dashboard] WARNING: could not write ledger {path}: {exc}")
