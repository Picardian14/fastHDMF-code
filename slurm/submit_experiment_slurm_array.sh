#!/bin/bash

set -euo pipefail

# Determine experiments directory (allow choosing any sub‐directory under ./configs)
declare -a DIR_CHOICES=()
# 1) if ./experiments exists, add it as an option
if [ -d "experiments" ]; then
    DIR_CHOICES+=("experiments")
fi
# 2) gather all subdirs under ./configs
if [ -d "configs" ]; then
    while IFS= read -r sub; do
        DIR_CHOICES+=("configs/$sub")
    done < <(find configs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

if [ ${#DIR_CHOICES[@]} -eq 0 ]; then
    echo "No valid experiment directories found (./experiments or subdirs under ./configs)." >&2
    exit 1
fi

echo "Available experiment directories:"
for i in "${!DIR_CHOICES[@]}"; do
    echo "  $((i+1))) ${DIR_CHOICES[$i]}"
done
echo

read -rp "Select directory by number or type path: " CHOICE
if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#DIR_CHOICES[@]}" ]; then
    EXPERIMENTS_DIR="${DIR_CHOICES[$((CHOICE-1))]}"
else
    EXPERIMENTS_DIR="$CHOICE"
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    if [ -d "configs/experiments" ]; then
        EXPERIMENTS_DIR="configs/experiments"
    fi
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Could not find experiments directory (looked for 'experiments' and 'configs/experiments')." >&2
    exit 1
fi

# Build list of experiment YAMLs
mapfile -t EXP_FILES < <(find "$EXPERIMENTS_DIR" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' | sort)

if [ ${#EXP_FILES[@]} -eq 0 ]; then
    echo "No experiment YAML files found in $EXPERIMENTS_DIR" >&2
    exit 1
fi

# Strip extensions for IDs
EXP_IDS=()
for f in "${EXP_FILES[@]}"; do
    EXP_IDS+=("${f%.yaml}")
done

print_menu() {
    echo "Available experiments (from $EXPERIMENTS_DIR):"
    local idx=1
    for id in "${EXP_IDS[@]}"; do
        echo "  $idx) $id"
        idx=$((idx+1))
    done
    echo
}

EXPERIMENT_INPUT="${1-}"

if [ -z "$EXPERIMENT_INPUT" ]; then
    print_menu
    read -rp "Select experiment by number or type experiment_id: " EXPERIMENT_INPUT
fi

# Decide if numeric selection or direct id
if [[ "$EXPERIMENT_INPUT" =~ ^[0-9]+$ ]]; then
    SEL_INDEX=$EXPERIMENT_INPUT
    if [ "$SEL_INDEX" -lt 1 ] || [ "$SEL_INDEX" -gt ${#EXP_IDS[@]} ]; then
        echo "Invalid selection index: $SEL_INDEX" >&2
        exit 1
    fi
    EXPERIMENT_ID="${EXP_IDS[$((SEL_INDEX-1))]}"
else
    EXPERIMENT_ID="$EXPERIMENT_INPUT"
    # Validate it exists
    if [[ ! " ${EXP_IDS[*]} " =~ [[:space:]]${EXPERIMENT_ID}[[:space:]] ]]; then
        echo "Experiment id '$EXPERIMENT_ID' not found among available IDs." >&2
        echo
        print_menu
        exit 1
    fi
fi

# Add day-month-hour-minute tier to experiment ID to ensure uniqueness
#TIMESTAMP=$(date +"%d%m%H%M")
#EXPERIMENT_ID="${EXPERIMENT_ID}_$TIMESTAMP"

CONFIG_FILENAME="${EXPERIMENTS_DIR}/${EXPERIMENT_ID}.yaml"

echo "Submitting SLURM array job for experiment: $EXPERIMENT_ID (config: $CONFIG_FILENAME)"

# Calculate grid size and suggest array size
echo "Calculating grid size..."

GRID_INFO=$(python fastHDMF/utils/calculate_grid_size.py "$CONFIG_FILENAME")
echo "$GRID_INFO"

# Extract grid size from output
GRID_SIZE=$(echo "$GRID_INFO" | grep "Total grid combinations:" | awk '{print $4}')
SUGGESTED_JOBS=$(echo "$GRID_INFO" | grep "Suggested SLURM array size:" | awk '{print $5}')

echo
echo "Grid size: $GRID_SIZE combinations"
echo "Suggested array size: $SUGGESTED_JOBS jobs"
echo

# Ask user for number of jobs
read -rp "Enter number of SLURM array jobs (or press Enter for suggested: $SUGGESTED_JOBS): " USER_JOBS

if [ -z "$USER_JOBS" ]; then
    ARRAY_SIZE=$SUGGESTED_JOBS
else
    ARRAY_SIZE=$USER_JOBS
fi

# Validate array size
if ! [[ "$ARRAY_SIZE" =~ ^[0-9]+$ ]] || [ "$ARRAY_SIZE" -lt 1 ]; then
    echo "Invalid array size: $ARRAY_SIZE" >&2
    exit 1
fi

if [ "$ARRAY_SIZE" -gt "$GRID_SIZE" ]; then
    echo "Warning: Array size ($ARRAY_SIZE) is larger than grid size ($GRID_SIZE). Setting to grid size."
    ARRAY_SIZE=$GRID_SIZE
fi

echo "Using array size: $ARRAY_SIZE (0-$((ARRAY_SIZE-1)))"

# Ask user for number of CPUs per task
read -rp "Enter number of CPUs per task (or press Enter for default: 1): " USER_CPUS
if [ -z "$USER_CPUS" ]; then
    CPUS=1
else
    CPUS=$USER_CPUS
fi
# Validate CPUs per task
if ! [[ "$CPUS" =~ ^[0-9]+$ ]] || [ "$CPUS" -lt 1 ]; then
    echo "Invalid CPUs per task: $CPUS" >&2
    exit 1
fi

echo "Using CPUs per task: $CPUS"

# Ask user for memory per task
read -rp "Enter memory per task in GB (or press Enter for default: 16GB - works for 32 cores on 100 regions): " USER_MEMORY
if [ -z "$USER_MEMORY" ]; then
    MEMORY="16G"
else
    # Validate memory input and add G suffix if not present
    if [[ "$USER_MEMORY" =~ ^[0-9]+$ ]]; then
        MEMORY="${USER_MEMORY}G"
    elif [[ "$USER_MEMORY" =~ ^[0-9]+[GM]$ ]]; then
        MEMORY="$USER_MEMORY"
    else
        echo "Invalid memory format. Use format like: 16, 16G, or 16000M" >&2
        exit 1
    fi
fi

echo "Using memory per task: $MEMORY"

echo
echo "Final configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Config file: $CONFIG_FILENAME"
echo "  Grid size: $GRID_SIZE"
echo "  Array size: $ARRAY_SIZE"
echo "  CPUs per task: $CPUS"
echo "  Memory per task: $MEMORY"
echo

# Ask for CPU override
read -rp "Enter number of CPUs per task (or press Enter for default: $CPUS): " USER_CPUS_OVERRIDE
if [ -n "$USER_CPUS_OVERRIDE" ]; then
    CPUS=$USER_CPUS_OVERRIDE
fi

# Submit the array job and capture the job ID
JOB_ID=$(sbatch --parsable --cpus-per-task=$CPUS --mem=$MEMORY --job-name=$EXPERIMENT_ID --array=0-$((ARRAY_SIZE-1)) slurm/run_experiment_array.sh "$EXPERIMENT_ID" "$ARRAY_SIZE" "$CPUS")

echo "Array job submitted with ID: $JOB_ID"

# Submit the integration job with dependency on the array job
INTEGRATION_JOB_ID=$(sbatch --job-name=$EXPERIMENT_ID --parsable --dependency=afterok:$JOB_ID slurm/integrate_results.sh "$EXPERIMENT_ID")

echo "Integration job submitted with ID: $INTEGRATION_JOB_ID"
echo "Integration will run after all array jobs complete"
