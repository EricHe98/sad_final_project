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

pyscript=/home/abh466/sad_final_project/src/models/MultVAE/gen_user_latent.py
model_dir=/scratch/abh466/sad_data/models/multVAE/
model_run_id=50fbe8aa06e24238890d1ab10fdf2c1d 
epoch=199 
train_dir=/scratch/abh466/sad_data/processed/full/test/user_to_queries.pkl
hotel_dir=/scratch/abh466/sad_data/processed/hotel_hash.json
user_dir=/scratch/abh466/sad_data/processed/user_hash.json
output_dir=/scratch/abh466/sad_data/processed/full/test/
python3 $pyscript -m $model_dir -n $model_run_id -e $epoch -d $train_dir -i $hotel_dir -u $user_dir -o $output_dir
