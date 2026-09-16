#!/bin/bash
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/
#SBATCH --output=slurm/outputs/HDMF_%A.out
#SBATCH --error=slurm/errors/HDMF_%A.err

# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Positional arguments: EXPERIMENT_ID, CPUS
EXPERIMENT_ID="$1"
CPUS="$2"

if [ -z "$EXPERIMENT_ID" ]; then
	echo "Usage: $0 <EXPERIMENT_ID> [CPUS]" >&2
	exit 1
fi

if [ -z "$CPUS" ]; then
	CPUS=1
fi

# Run Python script inside Singularity container with cpus (single-node run)
singularity exec "$SINGULARITY_IMAGE" bash -c "source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && python3 fastHDMF/run_experiment.py $EXPERIMENT_ID --cpus $CPUS"