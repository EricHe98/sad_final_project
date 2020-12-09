#!/bin/bash

#SBATCH --job-name=val_metrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=awd275@nyu.edu
#SBATCH --output=slurm_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

source ~/.bashrc
conda activate dsga3001

cd /home/awd275/Search_and_Discovery/sad_final_project/



export PYTHONPATH="./"

# Execute the script
py3=/home/awd275/miniconda3/envs/dsga3001/bin/python3
pyscript=/home/awd275/Search_and_Discovery/sad_final_project/src/models/MultVAE/val_metrics.py
model_folder=/scratch/work/js11133/sad_data/models/multVAE/
model_run_id=5dbd76b2cf4a48bea6164d32d8ec66c7_annealed
max_epoch=48
dataset_pkl=/scratch/work/js11133/sad_data/processed/full/val/user_to_queries.pkl
user_hash=/scratch/work/js11133/sad_data/processed/user_hash.json
hotel_hash=/scratch/work/js11133/sad_data/processed/hotel_hash.json
output_dir=/scratch/work/js11133/sad_data/predictions/
$py3 $pyscript -m $model_folder -n $model_run_id -e $max_epoch -d $dataset_pkl -i $hotel_hash -u $user_hash -o $output_dir &> output3.txt





