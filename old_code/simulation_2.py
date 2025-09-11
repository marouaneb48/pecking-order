
from utils import BL, CF_BL, oracle_bl, oracle_CF_BL

import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from joblib import Parallel, delayed




# Fixed parameters
N_e, N_b = 1000, 1000
M_e, M_b = 100, 100
K = 600
p = 1
c = 0.6
t = 0.08
rf = 0.05

# Parameter grids
theta_oracle_list = np.linspace(0.01, 0.99, 10)
theta_e_list = np.linspace(0.01, 0.99, 10)
theta_b_list = np.linspace(0.01, 0.99, 10)
precision_e_list = np.arange(1, 1000, 100)
precision_b_list = np.arange(1, 1000, 100)

# Build parameter grid
param_grid = [
    (to, te, tb, pe, pb)
    for to in theta_oracle_list
    for te in theta_e_list
    for tb in theta_b_list
    for pe in precision_e_list
    for pb in precision_b_list
]


# Get job index and total number of chunks
array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_chunks = int(os.environ.get("NUM_ARRAY_TASKS", 10))  # Set this in your SLURM script

# Slice the grid
chunk_size = len(param_grid) // num_chunks
start = array_index * chunk_size
end = len(param_grid) if array_index == num_chunks - 1 else (array_index + 1) * chunk_size
param_chunk = param_grid[start:end]



os.makedirs("checkpoints", exist_ok=True)


checkpoint_file = "results_checkpoint_{array_index}.pkl"
final_results_file = "final_results.csv"
results = []

# Simulation logic
def compute_profits(theta_oracle_i, theta_e_i, theta_b_i, precision_e_i, precision_b_i):
    CF_BL_value = CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b,
                        theta_e_i, theta_b_i, precision_e_i, precision_b_i).mean_profit()
    BL_value = BL(K, p, c, rf, M_e, N_e, M_b, N_b,
                  theta_e_i, theta_b_i, precision_e_i, precision_b_i).mean_profit()

    oracle_bl_value = oracle_bl(K, p, c, rf, M_e, N_e, M_b, N_b,
                                theta_oracle_i, theta_b_i, precision_b_i).mean_profit()
    
    oracle_cf_bl_value = oracle_CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b,
                                      theta_oracle_i, theta_b_i, precision_b_i).mean_profit()

    return {
        'theta_oracle': theta_oracle_i,
        'theta_e': theta_e_i,
        'theta_b': theta_b_i,
        'precision_e': precision_e_i,
        'precision_b': precision_b_i,
        'CF_BL': CF_BL_value,
        'BL': BL_value,
        'oracle_bl': oracle_bl_value,
        'oracle_cf_bl': oracle_cf_bl_value
    }



# Load checkpoint
completed_indices = set()
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
        results = checkpoint_data['results']
        completed_indices = set(checkpoint_data['completed_indices'])
    print(f"Resuming from checkpoint. {len(completed_indices)} parameter combinations already computed.")

# Tracker
def compute_and_track(idx, to, te, tb, pe, pb):
    if idx in completed_indices:
        return None
    res = compute_profits(to, te, tb, pe, pb)
    return idx, res

# Execute in chunks
import multiprocessing

# Get number of CPUs from SLURM or fall back to all
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

chunk_size = 100
results_dict = {
    (r['theta_oracle'], r['theta_e'], r['theta_b'], r['precision_e'], r['precision_b']): r
    for r in results
}

with tqdm(total=len(param_grid)) as pbar:
    pbar.update(len(completed_indices))

    for chunk_start in range(0, len(param_chunk), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(param_chunk))
        chunk = param_chunk[chunk_start:chunk_end]

        processed = Parallel(n_jobs=num_cores)(
            delayed(compute_and_track)(start + chunk_start + i, to, te, tb, pe, pb)
            for i, (to, te, tb, pe, pb) in enumerate(chunk)
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

        # Save checkpoint
        results = list(results_dict.values())
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'results': results, 'completed_indices': completed_indices}, f)
        print(f"Checkpoint saved after {len(completed_indices)} combinations.")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(final_results_file, index=False)

# Remove checkpoint file
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print(f"Computation complete. Results saved to '{final_results_file}'")
