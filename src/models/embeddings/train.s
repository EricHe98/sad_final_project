#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=185GB
#SBATCH --job-name=sad_embeddings
#SBATCH --mail-type=END
#SBATCH --mail-user=js11133@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=/scratch/js11133/sad_final_project
cd $SRCDIR

# Activate pipenv shell
pipenv shell
pip install -e .

# Execute the script
python src/models/embeddings/train.py full --split train --emb /scratch/work/js11133/sad_data/embeddings/full/embeddings.parquet --data /scratch/work/js11133/sad_data/
# And we're done!


