#!/bin/bash

#SBATCH --job-name=prep_small_100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=24:00:00


module purge
conda init bash
source dsga3001

py3=/home/awd275/miniconda3/envs/dsga3001/bin/python3 
pyscript=/home/awd275/Search_and_Discovery/sad_final_project/src/data/data_preprocessing_multvae.py 

input1=/scratch/work/js11133/sad_data/processed
input2=/scratch/work/js11133/sad_data/raw/small_100/train
input3=/scratch/work/js11133/sad_data/processed/small_100/train

$py3 $pyscript $input1 $input2 $input3 &> output_small_train.txt
input1=/scratch/work/js11133/sad_data/processed
input2=/scratch/work/js11133/sad_data/raw/small_100/val
input3=/scratch/work/js11133/sad_data/processed/small_100/val

$py3 $pyscript $input1 $input2 $input3 &> output_small_val.txt

input1=/scratch/work/js11133/sad_data/processed
input2=/scratch/work/js11133/sad_data/raw/small_100/test
input3=/scratch/work/js11133/sad_data/processed/small_100/test

$py3 $pyscript $input1 $input2 $input3 &> output_small_test.txt



echo "done"

