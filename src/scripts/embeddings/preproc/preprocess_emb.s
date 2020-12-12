#!/bin/bash
#SBATCH --job-name=emb_lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

# Remove all unused system modules
module purge
module load pyspark/2.4.7
source activate sad_project_env

# Move into the directory that contains our code
SRCDIR=/home/js11133/sad_final_project
cd $SRCDIR

export PYTHONPATH="./"

python -u src/data/preprocessing_embeddings.py full --split train --data /scratch/abh466/sad_data/raw --out_dir $1 --in_samples $2 --out_samples $3

