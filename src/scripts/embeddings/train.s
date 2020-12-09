#!/bin/bash
#SBATCH --job-name=emb_lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=24:00:00

# Remove all unused system modules
module purge
source activate sad_project_env

# Move into the directory that contains our code
SRCDIR=/home/js11133/sad_final_project
cd $SRCDIR

export PYTHONPATH="./"

python -u src/models/embeddings/train.py full --split train --emb /scratch/abh466/sad_data/embeddings/full/embeddings.parquet --data /scratch/abh466/sad_data/raw

