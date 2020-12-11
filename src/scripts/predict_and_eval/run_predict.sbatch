#!/bin/bash

#SBATCH --job-name=predict_on_multvae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
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
py3=/scratch/awd275/miniconda3/envs/sad_project_env/bin/python3
pyscript=/home/awd275/sad_final_project/src/models/MultVAE/predict.py
run_id='abcd'
model_path=/scratch/abh466/sad_data/models/multVAE/multvae_ab29fa3a1694405b842cf84e51866918_annealed_epoch_626.uri
dataset_pkl=/scratch/abh466/sad_data/processed/full/test/user_to_queries.pkl
user_hash=/scratch/abh466/sad_data/processed/user_hash.json
hotel_hash=/scratch/abh466/sad_data/processed/hotel_hash.json
output_dir=/scratch/abh466/sad_data/predictions/
$py3 $pyscript -m $model_path -r $run_id -d $dataset_pkl -i $hotel_hash -u $user_hash -o $output_dir &> output9.txt





