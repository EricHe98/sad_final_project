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

pyscript=/home/abh466/sad_final_project/src/models/MultVAE/train.py
train_dir=/scratch/work/js11133/sad_data/processed/full/train/user_to_queries.pkl
val_dir=/scratch/work/js11133/sad_data/processed/full/val/user_to_queries.pkl
dict_dir=/scratch/work/js11133/sad_data/processed/hotel_hash.json
save_path=/scratch/work/js11133/sad_data/models/multVAE/
num_layers=1
learning_rate=1e-3
hidden=600
latent=200
python3 $pyscript -tr $train_dir -v $val_dir -d $dict_dir -s $save_path -l $num_layers -lr $learning_rate -hd $hidden -lt $latent	
