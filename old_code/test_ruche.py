import os 

num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

print(f"Using {num_cores} cores for parallel processing.")
