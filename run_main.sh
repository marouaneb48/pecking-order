#!/bin/bash
#SBATCH --job-name=dask_test
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:15:00
#SBATCH --output=dask_test_%j.out
#SBATCH --error=dask_test_%j.err


module load anaconda3/2024.06/gcc-13.2.0

source activate pecking_order

# Set environment variables for Dask
export DASK_DISTRIBUTED__WORKER__DAEMON=False
export DASK_DISTRIBUTED__ADMIN__TICK__LIMIT=3600s

echo "Starting Dask test on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Run the Dask test
python main.py

echo "Dask test completed at $(date)"