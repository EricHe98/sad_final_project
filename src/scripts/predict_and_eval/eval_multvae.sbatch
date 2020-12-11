#!/bin/bash

#SBATCH --job-name=predict_on_multvae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=awd275@nyu.edu
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate sad_project_env

cd /home/awd275/sad_final_project/



export PYTHONPATH="./"

# Execute the script
py3=python3
pyscript=/home/awd275/sad_final_project/src/eval/eval.py
$py3 $pyscript  &> output_eval asdsa full test





