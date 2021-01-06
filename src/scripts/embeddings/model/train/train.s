#!/bin/bash
#SBATCH --job-name=emb_lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb
#SBATCH --time=48:00:00

# Remove all unused system modules
module purge
source activate sad_project_env

# Move into the directory that contains our code
SRCDIR=/home/js11133/sad_final_project
cd $SRCDIR

export PYTHONPATH="./"

python src/models/embeddings/train.py full --split train --emb $1/embeddings.parquet --data /scratch/abh466/sad_data/raw

