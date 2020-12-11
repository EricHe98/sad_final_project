#!/bin/bash

#SBATCH --job-name=predict_lambdamart_user
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=8:00:00
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
export PYTHONPATH="./"

pyscript=/home/abh466/sad_final_project/src/models/lambdamart_user_latent/predict.py
run_id=cc7ecd02ec004c53811da66e9e951071
test_dir=/scratch/work/js11133/sad_data/raw/full/test
latent_dir=/scratch/work/js11133/sad_data/processed/full/test/user_latent_test.parquet
split=test
python3 -u $pyscript -r $run_id -d $test_dir -l $latent_dir -s $split
