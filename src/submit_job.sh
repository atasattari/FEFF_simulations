#!/bin/bash
#SBATCH --job-name=FEFF
#SBATCH --output= fill in 
#SBATCH --error= fill in 
#SBATCH --account= fill in  
#SBATCH --time=2:59:00
#SBATCH --mail-user= fill in
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3000MB

module load scdms/V04-14
singularity exec $SCDMS_IMAGE python3 {fill in}/src/FEFF.py -n $SLURM_CPUS_PER_TASK
