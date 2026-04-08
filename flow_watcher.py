#!/usr/bin/env python3
"""
flow_watcher.py — Live ns-3 flow CSV watcher + batched inference runner
========================================================================
Watches a growing flows CSV (produced by ns-3 / pcap_flow_extractor.py),
batches every N new data rows, writes them to prototype/input/flows.csv,
calls inference.py --mode a, then saves the output as a sequentially
numbered file (output/predictions_1.csv, output/predictions_2.csv, …).

Usage
-----
    # From prototype/ directory:
    python flow_watcher.py

    # Full options:
    python flow_watcher.py \\
        --source    ../ns3-ddos/scratch/flows.csv \\
        --batch-size 5 \\
        --poll-interval 1.0 \\
        --prototype-dir .

Arguments
---------
--source        Path to the ns-3 output flows CSV (watched file).
                Default: ../ns3-ddos/scratch/flows.csv
--batch-size    Number of NEW data rows to accumulate before triggering
                one inference run.  Default: 5
--poll-interval Seconds between file-size checks.  Default: 1.0
--prototype-dir Root of the prototype package (contains inference.py).
                Default: directory of this script.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_args():
    p = argparse.ArgumentParser(description="Live ns-3 flow watcher + batched inference")
    p.add_argument(
        "--source",
        default=os.path.join(_HERE, "..", "ns3-ddos", "scratch", "flows.csv"),
        help="Path to the growing ns-3 flows CSV (default: ../ns3-ddos/scratch/flows.csv)",
    )
    p.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of new data rows to accumulate before one inference run (default: 5)",
    )
    p.add_argument(
        "--poll-interval", type=float, default=1.0,
        help="Seconds between file-size checks (default: 1.0)",
    )
    p.add_argument(
        "--prototype-dir", default=_HERE,
        help="Root of the prototype package — must contain inference.py (default: script dir)",
    )
    return p.parse_args()


def _read_new_rows(source_path: str, rows_seen: int):
    """
    Read `source_path`, skip the header + rows_seen data rows, return
    (header, list-of-new-rows, total-data-rows-now).
    Returns (None, [], rows_seen) if the file doesn't exist yet.
    """
    if not os.path.exists(source_path):
        return None, [], rows_seen

    with open(source_path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        return None, [], rows_seen

    header = rows[0]
    data   = rows[1:]          # skip CSV header line
    new    = data[rows_seen:]  # rows we haven't seen yet
    return header, new, len(data)


def _write_input_csv(prototype_dir: str, header, rows):
    """Overwrite prototype/input/flows.csv with header + rows."""
    input_path = os.path.join(prototype_dir, "input", "flows.csv")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    with open(input_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return input_path


def _run_inference(prototype_dir: str):
    """Invoke inference.py --mode a as a subprocess from prototype_dir."""
    result = subprocess.run(
        [sys.executable, "inference.py", "--mode", "a"],
        cwd=prototype_dir,
    )
    if result.returncode != 0:
        print(f"[watcher] WARNING: inference.py exited with code {result.returncode}")


def _archive_output(prototype_dir: str, batch_num: int):
    """
    Move output/predictions.csv → output/predictions_{batch_num}.csv.
    If the file doesn't exist (inference failed silently), skip.
    """
    out_dir  = os.path.join(prototype_dir, "output")
    src      = os.path.join(out_dir, "predictions.csv")
    dst      = os.path.join(out_dir, f"predictions_{batch_num}.csv")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"[watcher] Saved → {dst}")
    else:
        print(f"[watcher] WARNING: predictions.csv not found after batch {batch_num}; skipping archive.")


def main():
    args = parse_args = _parse_args()

    source_path   = os.path.abspath(args.source)
    batch_size    = args.batch_size
    poll_interval = args.poll_interval
    proto_dir     = os.path.abspath(args.prototype_dir)

    print(f"[watcher] Source      : {source_path}")
    print(f"[watcher] Batch size  : {batch_size} rows")
    print(f"[watcher] Poll every  : {poll_interval}s")
    print(f"[watcher] Prototype   : {proto_dir}")
    print("[watcher] Press Ctrl-C to stop.\n")

    rows_seen        = 0     # number of data rows already dispatched to inference
    batch_num        = 0     # sequential counter for output files
    pending          = []    # buffered rows not yet dispatched
    header           = None
    last_new_row_at  = None  # time.monotonic() when we last saw a new row
    IDLE_FLUSH_SECS  = 5.0   # flush pending rows after this many seconds of no new rows

    def _flush(reason: str):
        nonlocal pending, rows_seen, batch_num
        if not pending:
            return
        batch_num += 1
        n = len(pending)
        rows_seen += n
        print(f"\n[watcher] Batch {batch_num} ({reason}): dispatching {n} rows "
              f"(total rows seen so far: {rows_seen})")
        _write_input_csv(proto_dir, header, pending)
        pending = []
        _run_inference(proto_dir)
        _archive_output(proto_dir, batch_num)

    try:
        while True:
            hdr, new_rows, total = _read_new_rows(source_path, rows_seen + len(pending))

            if hdr is not None:
                if header is None:
                    header = hdr
                if new_rows:
                    last_new_row_at = time.monotonic()
                    pending.extend(new_rows)

            # Fire as many complete batches as we have
            while len(pending) >= batch_size:
                batch      = pending[:batch_size]
                pending    = pending[batch_size:]
                rows_seen += batch_size
                batch_num += 1

                print(f"\n[watcher] Batch {batch_num}: dispatching {len(batch)} rows "
                      f"(total rows seen so far: {rows_seen})")

                _write_input_csv(proto_dir, header, batch)
                _run_inference(proto_dir)
                _archive_output(proto_dir, batch_num)

            # Flush partial batch if no new rows have arrived in IDLE_FLUSH_SECS
            if (pending and last_new_row_at is not None
                    and time.monotonic() - last_new_row_at >= IDLE_FLUSH_SECS):
                _flush(f"idle >{IDLE_FLUSH_SECS:.0f}s")
                last_new_row_at = None  # reset so we don't re-flush on next tick

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        leftover = len(pending)
        print(f"\n[watcher] Stopped. {leftover} buffered row(s) not yet dispatched "
              f"(need {batch_size - leftover} more to complete next batch).")
        if leftover:
            ans = input("[watcher] Run inference on the partial batch now? [y/N] ").strip().lower()
            if ans == "y":
                _flush("manual flush on exit")


if __name__ == "__main__":
    main()
