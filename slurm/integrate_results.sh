#!/bin/bash
#SBATCH --partition=medium
#SBATCH --job-name=integrate_results
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/
#SBATCH --output=slurm/outputs/INTEGRATE_%j.out
#SBATCH --error=slurm/errors/INTEGRATE_%j.err

# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif
EXPERIMENT_ID=$1

echo "Integrating results for experiment: $EXPERIMENT_ID"

# Run integration script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c "source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && python3 fastHDMF/integrate_distributed_results.py $EXPERIMENT_ID"

echo "Integration completed"
