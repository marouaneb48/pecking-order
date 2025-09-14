import logging, sys, traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def _excepthook(exc_type, exc, tb):
    print("Uncaught exception:", file=sys.stderr)
    traceback.print_exception(exc_type, exc, tb, file=sys.stderr)

sys.excepthook = _excepthook


from utils.utils_3 import BL, CF_BL
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
import numpy as np
import pandas as pd
from itertools import product
import os
import csv
import json
import signal
from datetime import datetime


CHECKPOINT_CSV = "results/final_results_3.checkpoint.csv"
FINAL_CSV      = "results/final_results_3.csv"
FINAL_PARQUET  = "results/final_results_3.parquet"
MANIFEST_JSON  = "results/final_results_3.manifest.json"
FLUSH_EVERY    = 500          # write to checkpoint every N results
MAX_WORKERS    = 100          # cap on Dask jobs


GRID = {
    # Model size
    "N_e": [1000],                # e.g. [500, 1000, 2000]
    "N_b": [1000],
    "M_e": [100],
    "M_b": [100],

    # Economic parameters
    "K":  [600],                  # e.g. [400, 600, 800]
    "p":  [1],                    # e.g. [0.5, 0.75, 1.0]
    "c":  [0.6],                  # e.g. [0.2, 0.4, 0.6]
    "t":  [0.08],                 # e.g. [0.02, 0.05, 0.08]
    "rf": [0.05],                 # e.g. [0.01, 0.03, 0.05]

    # Behavioral parameters
    "theta_e": list(np.linspace(0.05, 0.95, 19)),
    "theta_b": list(np.linspace(0.05, 0.95, 19)),
    "precision_e": [10, 100, 1000],
    "precision_b": [10, 100, 1000],
}


KEYS = list(GRID.keys())
VALUE_LISTS = [GRID[k] for k in KEYS]

def total_jobs(value_lists):
    tot = 1
    for vs in value_lists:
        tot *= len(vs)
    return tot

TOTAL_JOBS = total_jobs(VALUE_LISTS)
print(f"Total combinations in grid: {TOTAL_JOBS:,}")

def read_checkpoint_keys(path):
    if not os.path.exists(path):
        return set()
    # Read minimal columns to reduce memory; CSV is friendly for incremental appends
    usecols = KEYS  # keys uniquely identify a row
    try:
        df = pd.read_csv(path, usecols=usecols)
    except pd.errors.EmptyDataError:
        return set()
    # Build tuple keys
    return set(tuple(row[k] for k in KEYS) for _, row in df.iterrows())


DONE_KEYS = read_checkpoint_keys(CHECKPOINT_CSV)
print(f"Already completed from checkpoint: {len(DONE_KEYS):,}")


def combos_to_run():
    for vals in product(*VALUE_LISTS):
        params = dict(zip(KEYS, vals))
        key_tuple = tuple(params[k] for k in KEYS)
        if key_tuple not in DONE_KEYS:
            yield params

REMAINING_JOBS = sum(1 for _ in combos_to_run())
if REMAINING_JOBS == 0:
    print("Nothing to do—grid fully completed in checkpoint.")
else:
    print(f"Remaining combos to compute: {REMAINING_JOBS:,}")


# Write a manifest (useful for provenance / audit)
manifest = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "grid": {k: list(map(float, v)) if isinstance(v, np.ndarray) else v for k, v in GRID.items()},
    "keys": KEYS,
    "total_jobs": TOTAL_JOBS,
}

with open(MANIFEST_JSON, "w") as f:
    json.dump(manifest, f, indent=2)



if REMAINING_JOBS == 0:
    # No need to spin a cluster if done; still produce final outputs below
    client = None
    cluster = None
else:
    cluster = SLURMCluster(
        cores=1,
        queue="cpu_med",
        memory="1GB",
        walltime="02:00:00",
        account="ilocos-umicrowd",
        log_directory="dask_logs",
        job_extra=["--signal=TERM@120"]
    )
    cluster.scale(jobs=min(REMAINING_JOBS, MAX_WORKERS))
    client = Client(cluster)
    print(f"Cluster scaled to ~{min(REMAINING_JOBS, MAX_WORKERS)} workers.")

def compute(params):
    N_e = params["N_e"]; N_b = params["N_b"]
    M_e = params["M_e"]; M_b = params["M_b"]
    K = params["K"]; p = params["p"]; c = params["c"]; t = params["t"]; rf = params["rf"]
    te = params["theta_e"]; tb = params["theta_b"]
    pe = params["precision_e"]; pb = params["precision_b"]

    return {
        **params,
        "CF_BL": CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b, te, tb, pe, pb).mean_profit(),
        "BL":    BL(K, p, c, rf, M_e, N_e, M_b, N_b, te, tb, pe, pb).mean_profit(),
    }

# -------------------------------
# 5) Safe buffered writer
# -------------------------------
_buffer = []
_written_header = os.path.exists(CHECKPOINT_CSV) and os.path.getsize(CHECKPOINT_CSV) > 0

def flush_buffer():
    global _buffer, _written_header
    if not _buffer:
        return
    # Append to CSV
    fieldnames = KEYS + ["CF_BL", "BL"]
    write_header = not _written_header
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            _written_header = True
        writer.writerows(_buffer)
    print(f"\nFlushed {_buffer_len()} rows to checkpoint ({CHECKPOINT_CSV}).")
    _buffer.clear()

def _buffer_len():
    return len(_buffer)

# Ensure we flush on SIGINT/SIGTERM
def _handle_signal(signum, frame):
    print(f"\nReceived signal {signum}. Flushing buffer before exit...")
    flush_buffer()
    # Let program terminate naturally (or re-raise KeyboardInterrupt)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# -------------------------------
# 6) Run (with resume)
# -------------------------------
completed = len(DONE_KEYS)

try:
    if REMAINING_JOBS > 0:
        # Submit in one shot (you can batch if memory becomes an issue)
        to_do = list(combos_to_run())
        futures = client.map(compute, to_do, pure=False, retries=2)

        for fut in as_completed(futures):
            res = fut.result()
            _buffer.append(res)
            completed += 1

            # Periodic flush
            if _buffer_len() >= FLUSH_EVERY:
                flush_buffer()

            # Progress line
            print(f"Progress: {completed}/{TOTAL_JOBS} ({completed / TOTAL_JOBS:.1%})", end="\r")

        # Final flush
        flush_buffer()
    else:
        print("Skipping compute phase; using existing checkpoint.")
finally:
    # Clean up cluster even on interruption
    if client is not None:
        client.close()
    if cluster is not None:
        cluster.close()

# -------------------------------
# 7) Produce final outputs (dedup)
# -------------------------------
# Merge from checkpoint, drop duplicates by key, and save final artifacts
if os.path.exists(CHECKPOINT_CSV) and os.path.getsize(CHECKPOINT_CSV) > 0:
    df = pd.read_csv(CHECKPOINT_CSV)
    # Drop duplicate key rows keeping last (in case of re-runs)
    df.sort_values(by=KEYS, inplace=True)
    df = df.drop_duplicates(subset=KEYS, keep="last")
    df.to_csv(FINAL_CSV, index=False)
    # Parquet is optional if pyarrow/fastparquet is installed
    try:
        df.to_parquet(FINAL_PARQUET, index=False)
        print(f"Saved {FINAL_CSV} and {FINAL_PARQUET}")
    except Exception as e:
        print(f"Saved {FINAL_CSV}. Parquet skipped: {e}")
else:
    print("No checkpoint data found; nothing to finalize.")