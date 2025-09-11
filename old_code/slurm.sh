#!/bin/bash

#SBATCH --job-name=hello_numpy
#SBATCH --output=%x.o%j 
#SBATCH --time=01:00:00 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=38  
#SBATCH --partition=cpu_med
#SBATCH --signal=TERM@60  
#SBATCH --requeue    



module load anaconda3/2024.06/gcc-13.2.0
source activate pecking_order
# Run python script
python simulation.py
