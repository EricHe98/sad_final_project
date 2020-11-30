#!/bin/bash

#SBATCH --job-name=train_multvae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=awd275@nyu.edu
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate dsga3001


cd /home/awd275/Search_and_Discovery/sad_final_project/src/models/MultVAE


# Execute the script
py3=/home/awd275/miniconda3/envs/dsga3001/bin/python3 
pyscript=/home/awd275/Search_and_Discovery/sad_final_project/src/models/MultVAE/train.py

$py3 $pyscript &> output_train.txt

