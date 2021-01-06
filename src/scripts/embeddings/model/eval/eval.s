#!/bin/bash
#SBATCH --job-name=emb_lm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00

# Remove all unused system modules
module purge
source activate sad_project_env

# Move into the directory that contains our code
SRCDIR=/home/js11133/sad_final_project
cd $SRCDIR

RUN_ID=32d6e30644b54bdeadc6b1e843a0fabe
export PYTHONPATH="./"
pip install mlflow==1.11.0

python -u src/eval/eval.py $RUN_ID full val --data /scratch/abh466/sad_data/raw

