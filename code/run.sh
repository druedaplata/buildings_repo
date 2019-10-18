#!/bin/bash

#SBATCH --partition=accel-2
#SBATCH --nodes=1                                                             
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --job-name=12d8
#SBATCH -o output/12d8.out      # File to which STDOUT will be written
#SBATCH -e output/12d8.err      # File to which STDERR will be written
##SBATCH --mail-type=ALL
##SBATCH --mail-user=ing.diegorueda@gmail.com

export SBATCH_EXPORT=NONE
export OMP_NUM_THREADS=1

module load cuda/9.0
module load python/3.5.2_intel-2017_update-1

source activate keras_intel

python train.py
