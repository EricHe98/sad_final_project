#!/bin/bash
#SBATCH --job-name=emb_lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Remove all unused system modules
module purge
source activate sad_project_env

# Move into the directory that contains our code
SRCDIR=/home/js11133/sad_final_project
cd $SRCDIR

export PYTHONPATH="./"

python -u src/models/embeddings/gen_emb.py --data $1 

