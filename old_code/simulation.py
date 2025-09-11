from utils import BL, CF_BL, oracle_bl, oracle_CF_BL

import numpy as np
import pandas as pd
import pickle
import os
import sys
import signal
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Fixed parameters
N_e, N_b = 1000, 1000
M_e, M_b = 100, 100
K = 600
p = 1
c = 0.6
t = 0.08
rf = 0.05

# Parameter grids
# theta_oracle_list = [0.3, 0.5, 0.8]
theta_e_list = np.linspace(0.1, 0.9, 9)
theta_b_list = np.linspace(0.1, 0.9, 9)
precision_e_list = [10, 100, 1000]
precision_b_list = [10, 100, 1000]

checkpoint_file = "results_checkpoint.pkl"
final_results_file = "final_results.csv"
results = []
completed_indices = set()

# Save checkpoint
def save_checkpoint():
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'results': results, 'completed_indices': completed_indices}, f)
    print(f"[Checkpoint] Saved {len(completed_indices)} combinations.")

# SIGTERM handler
def handle_sigterm(signum, frame):
    print("[SIGTERM] Received. Saving checkpoint before exit...")
    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# Simulation logic
def compute_profits(theta_oracle_i, theta_e_i, theta_b_i, precision_e_i, precision_b_i):
    CF_BL_value = CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b,
                        theta_e_i, theta_b_i, precision_e_i, precision_b_i).mean_profit()
    BL_value = BL(K, p, c, rf, M_e, N_e, M_b, N_b,
                  theta_e_i, theta_b_i, precision_e_i, precision_b_i).mean_profit()
    # oracle_bl_value = oracle_bl(K, p, c, rf, M_e, N_e, M_b, N_b,
    #                             theta_oracle_i, theta_b_i, precision_b_i).mean_profit()
    # oracle_cf_bl_value = oracle_CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b,
    #                                   theta_oracle_i, theta_b_i, precision_b_i).mean_profit()

    return {
        'theta_oracle': theta_oracle_i,
        'theta_e': theta_e_i,
        'theta_b': theta_b_i,
        'precision_e': precision_e_i,
        'precision_b': precision_b_i,
        'CF_BL': CF_BL_value,
        'BL': BL_value,
        # 'oracle_bl': oracle_bl_value,
        # 'oracle_cf_bl': oracle_cf_bl_value
    }

# Parameter grid
param_grid = [
    (_, te, tb, pe, pb)
    # for to in theta_oracle_list
    for te in theta_e_list
    for tb in theta_b_list
    for pe in precision_e_list
    for pb in precision_b_list
]

# Load checkpoint if it exists
if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            results = checkpoint_data['results']
            completed_indices = set(checkpoint_data['completed_indices'])
        print(f"[Resume] Loaded checkpoint with {len(completed_indices)} combinations.")
    except Exception as e:
        print(f"[Warning] Failed to load checkpoint: {e}")
        results = []
        completed_indices = set()

# Compute function with tracking
def compute_and_track(idx, _, te, tb, pe, pb):
    if idx in completed_indices:
        return None
    res = compute_profits(_, te, tb, pe, pb)
    return idx, res

# Dictionary for quick lookup
results_dict = {
    (_, r['theta_e'], r['theta_b'], r['precision_e'], r['precision_b']): r
    for r in results
}

# Parallel configuration
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
chunk_size = 100

# Processing loop
try:
    with tqdm(total=len(param_grid)) as pbar:
        pbar.update(len(completed_indices))

        for chunk_start in range(0, len(param_grid), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(param_grid))
            chunk = param_grid[chunk_start:chunk_end]

            processed = Parallel(n_jobs=num_cores)(
                delayed(compute_and_track)(chunk_start + i, _, te, tb, pe, pb)
                for i, (_, te, tb, pe, pb) in enumerate(chunk)
            )

            new_results = []
            for item in processed:
                if item is not None:
                    idx, res = item
                    key = (res['theta_oracle'], res['theta_e'], res['theta_b'], res['precision_e'], res['precision_b'])
                    results_dict[key] = res
                    completed_indices.add(idx)
                    new_results.append(res)

            pbar.update(len(new_results))
            results = list(results_dict.values())
            save_checkpoint()

except KeyboardInterrupt:
    print("[Interrupt] Ctrl+C received. Saving checkpoint and exiting...")
    save_checkpoint()
    sys.exit(0)

# Save final results
results_df = pd.DataFrame(list(results_dict.values()))
results_df.to_csv(final_results_file, index=False)

# Clean up
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print(f"[Done] Simulation complete. Results saved to '{final_results_file}'")
