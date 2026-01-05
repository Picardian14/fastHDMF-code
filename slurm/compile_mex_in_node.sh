#!/bin/bash
#SBATCH --partition=medium
#SBATCH --job-name=integrate_results
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/
#SBATCH --output=slurm/outputs/MEX_COMPILER%j.out
#SBATCH --error=slurm/errors/MEX_COMPILER%j.err

ml gcc
ml glibc
module unload FSL
ml glib
ml matlab/R2022b