#!/bin/bash

#SBATCH --job-name=hello_numpy
#SBATCH --output=%x.o%j 
#SBATCH --time=04:00:00 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32  
#SBATCH --partition=cpu_med



# Run python script
python test_ruche.py
