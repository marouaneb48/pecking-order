from utils_3 import BL, CF_BL
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
import numpy as np
import pandas as pd

# Parameters
N_e = N_b = 1000
M_e = M_b = 100
K, p, c, t, rf = 600, 1, 0.6, 0.08, 0.05
theta_e = theta_b = np.linspace(0.01, 0.99, 100)
precision_e = precision_b = [10, 100, 1000]

# Create cluster
cluster = SLURMCluster(cores=1, queue = "cpu_med", memory='1GB', walltime='02:00:00', account = "ilocos-umicrowd", log_directory="dask_logs")
cluster.scale(jobs=100)
client = Client(cluster)

# Compute function
def compute(te, tb, pe, pb):
    return {
        'theta_e': te, 'theta_b': tb, 'precision_e': pe, 'precision_b': pb,
        'CF_BL': CF_BL(K, p, c, t, rf, M_e, N_e, M_b, N_b, te, tb, pe, pb).mean_profit(),
        'BL': BL(K, p, c, rf, M_e, N_e, M_b, N_b, te, tb, pe, pb).mean_profit()
    }

# Submit all tasks
futures = [client.submit(compute, te, tb, pe, pb) 
           for te in theta_e for tb in theta_b 
           for pe in precision_e for pb in precision_b]

# Collect results with progress
results = []
for future in as_completed(futures):
    results.append(future.result())
    print(f"Progress: {len(results)}/{len(futures)}", end='\r')

# Save results
pd.DataFrame(results).to_csv('final_results_3.csv', index=False)
client.close()
cluster.close()