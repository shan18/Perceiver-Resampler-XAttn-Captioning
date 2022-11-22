#!/bin/bash
#SBATCH --job-name=mlslt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate mlslt
cd /scratch/rr3937/cv/Multilingual-Sign-Language-Translation

python run.py --num_workers 2 --epochs 40 --checkpoint_dir ckpt
