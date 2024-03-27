#!/bin/bash
#SBATCH --job-name='test'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=232GB
#SBATCH --output=test-%j.log
#SBATCH --time=24:00:00




date

python sparse_beam.py

date

