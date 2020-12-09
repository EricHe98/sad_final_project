#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dsga3001-lab0
#SBATCH --mail-type=END
#SBATCH --mail-user=eric.he@stern.nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
# module purge

# Move into the directory that contains our code
SRCDIR=/scratch/eh1885/sad_final_project

# Activate pipenv shell
pipenv shell
export PYTHONPATH="./"

# Execute the script
python src/models/xgbrfclassifier/train.py full 

# And we're done!

