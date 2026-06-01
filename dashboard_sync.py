"""
dashboard_sync.py -- back-fill the Sneaker Impact Dashboard from collected shoes.

Walks the incoming* folders, copies each crop into the dashboard's images/ folder,
and POSTs a record to the dashboard API. Re-runnable: a local ledger
(dashboard_synced.json) records what's already been pushed, so running it again
only sends new shoes. One dashboard batch is opened per source day-folder.

The dashboard must be running in APP_MODE=actual (same machine). See config.py
for DASHBOARD_URL / DASHBOARD_IMAGES_DIR / OPERATOR_ID.

Usage:
    python dashboard_sync.py                      # sync all incoming* folders
    python dashboard_sync.py --dry-run            # show what would be pushed
    python dashboard_sync.py --folder incoming06012026
    python dashboard_sync.py --url http://pi.local:8000
"""
import argparse
import json
import os
import sys

import config
from dashboard_client import DashboardClient
from dataset_utils import find_folders, load_entries

_DEFAULT_LEDGER = "dashboard_synced.json"


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
    try:
        with open(path, "w") as f:
            json.dump(ledger, f, indent=2)
    except Exception as exc:                       # noqa: BLE001 - non-fatal
        print(f"[sync] WARNING: could not write ledger {path}: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Back-fill the dashboard from collected shoes.")
    ap.add_argument("--folder", default=None, help="Single incoming* folder (default: all)")
    ap.add_argument("--root", default=config.OUTPUT_ROOT)
    ap.add_argument("--url", default=config.DASHBOARD_URL, help="dashboard base URL")
    ap.add_argument("--images-dir", default=config.DASHBOARD_IMAGES_DIR,
                    help="the dashboard's images/ folder on this machine")
    ap.add_argument("--operator", default=config.OPERATOR_ID)
    ap.add_argument("--ledger", default=_DEFAULT_LEDGER)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be pushed; make no changes")
    args = ap.parse_args()

    folders = ([os.path.join(args.root, args.folder)] if args.folder
               else find_folders(args.root))
    if not folders:
        print(f"No incoming* folders under '{args.root}'. Nothing to sync.")
        return

    ledger = load_ledger(args.ledger)
    client = DashboardClient(args.url, args.images_dir, args.operator)

    # Confirm the dashboard is up (and in the right mode) before pushing.
    if not args.dry_run:
        health = client.check()
        if health is None:
            print("[sync] dashboard not reachable -- start it in APP_MODE=actual "
                  "and check DASHBOARD_URL. Aborting.")
            sys.exit(1)
        if health.get("mode") != "actual":
            print(f"[sync] WARNING: dashboard is in '{health.get('mode')}' mode; "
                  "pushed shoes will mix with seeded fake data. "
                  "Set APP_MODE=actual and restart for clean data.")

    pushed = skipped = failed = 0

    for folder in folders:
        fname = os.path.basename(folder)
        entries = load_entries(folder)
        unsynced = [e for e in entries if e["jpg"] not in ledger["shoes"]]
        if not unsynced:
            continue
        print(f"\n=== {fname}: {len(unsynced)} new of {len(entries)} ===")

        # One dashboard batch per source folder; reuse it on later runs.
        batch_id = ledger["folders"].get(fname)
        if batch_id is None:
            if args.dry_run:
                batch_id = "(new batch)"
            else:
                batch_id = client.open_batch()
                if not batch_id:
                    print(f"[sync] could not open a batch for {fname}; skipping.")
                    failed += len(unsynced)
                    continue
                ledger["folders"][fname] = batch_id
                save_ledger(args.ledger, ledger)

        for e in unsynced:
            if args.dry_run:
                label = (e["meta"].get("classification") or "?")
                print(f"  would push {e['name']}  [{label}] -> {batch_id}")
                skipped += 1
                continue
            shoe_id = client.push_shoe(e["jpg"], e["meta"], batch_id)
            if shoe_id:
                ledger["shoes"][e["jpg"]] = shoe_id
                save_ledger(args.ledger, ledger)
                pushed += 1
            else:
                failed += 1

    print()
    if args.dry_run:
        print(f"Dry run: {skipped} shoe(s) would be pushed. No changes made.")
    else:
        print(f"Done. Pushed {pushed}, failed {failed}. "
              f"(Already-synced shoes were skipped.)")


if __name__ == "__main__":
    main()
