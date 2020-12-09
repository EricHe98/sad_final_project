#!/bin/bash

#SBATCH --job-name=train_multvae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abh466@nyu.edu
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate sad_project_env

# Execute the script

pyscript=/home/abh466/sad_final_project/src/models/lambdamart_user_latent/train.py
train_dir=/scratch/abh466/sad_data/raw/full/train
latent_dir=/scratch/abh466/sad_data/processed/full/train/train_emb.parquet
python3 -u $pyscript -d $train_dir -l $latent_dir
