#!/bin/bash

#SBATCH --job-name=val_metrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
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
py3=/scratch/awd275/miniconda3/envs/sad_project_env/bin/python3
pyscript=/home/awd275/sad_final_project/src/models/MultVAE/val_metrics.py
model_folder=/scratch/abh466/sad_data/models/multVAE/
model_run_id=50fbe8aa06e24238890d1ab10fdf2c1d_annealed
max_epoch=200
dataset_pkl=/scratch/abh466/sad_data/processed/full/val/user_to_queries.pkl
user_hash=/scratch/abh466/sad_data/processed/user_hash.json
hotel_hash=/scratch/abh466/sad_data/processed/hotel_hash.json
output_dir=/scratch/abh466/sad_data/predictions/
$py3 $pyscript -m $model_folder -n $model_run_id -e $max_epoch -d $dataset_pkl -i $hotel_hash -u $user_hash -o $output_dir &> output7.txt





