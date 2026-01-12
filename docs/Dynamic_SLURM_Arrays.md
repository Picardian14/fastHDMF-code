# Dynamic SLURM Array Configuration

This update adds the ability to automatically calculate grid sizes and configure SLURM array job counts based on the experiment configuration.

## New Features

### 1. Custom Parameter Generation Functions

You can now specify custom functions for generating parameter values in your config files:

```yaml
grid:
  # Traditional linear spacing
  lrj: {start: 1, end: 2.5, step: 0.125}
  
  # Logarithmic spacing
  G:
    fun: "np.logspace"
    args: [-1, 1, 20]  # 10^-1 to 10^1, 20 points
  
  # Linear spacing with exact number of points
  alpha:
    fun: "np.linspace"
    args: [0.1, 2.0]
    kwargs: {num: 25, endpoint: true}
```

### 2. Automatic Grid Size Calculation

The `utils/calculate_grid_size.py` utility calculates the total number of parameter combinations:

```bash
python utils/calculate_grid_size.py configs/experiments/Homeostatic_Grid.yaml
```

Output:
```
lrj: 4 values
taoj: 4 values

Total grid combinations: 16
Suggested SLURM array size: 5
Tasks per job: 3 (remainder: 1)
```

### 3. Interactive SLURM Array Configuration

The submission script now asks for the number of array jobs:

```bash
./slurm/submit_experiment_slurm_array.sh
```

Example interaction:
```
Available experiments (from configs/experiments):
  1) Homeostatic_Grid
  2) Another_Experiment

Select experiment by number or type experiment_id: 1

Calculating grid size...
lrj: 4 values
taoj: 4 values

Total grid combinations: 16
Suggested SLURM array size: 5

Grid size: 16 combinations
Suggested array size: 5 jobs

Enter number of SLURM array jobs (or press Enter for suggested: 5): 8

Using array size: 8 (0-7)
Array job submitted with ID: 12345
```

## How It Works

1. **Grid Size Calculation**: The system evaluates each parameter specification (traditional or custom function) to determine the number of values.

2. **Load Balancing**: The total grid is divided across the specified number of SLURM array jobs using contiguous blocks.

3. **Dynamic Array Submission**: The SLURM array size is set dynamically using `--array=0-N` where N is user-specified.

## Supported Functions

Currently supports any NumPy function via the `np.` prefix:
- `np.logspace(start, stop, num)`
- `np.linspace(start, stop, num, endpoint=True/False)`
- `np.arange(start, stop, step)`
- `np.geomspace(start, stop, num)`
- And any other NumPy array generation function

## Configuration Examples

### Traditional Grid
```yaml
grid:
  G: {start: 0.5, end: 3.0, step: 0.1}  # 25 values
  alpha: {start: 0.1, end: 2.0, step: 0.1}  # 19 values
# Total: 25 × 19 = 475 combinations
```

### Mixed Grid with Custom Functions
```yaml
grid:
  G:
    fun: "np.logspace"
    args: [-1, 1, 20]  # 20 values, logarithmic
  alpha: {start: 0.1, end: 2.0, step: 0.1}  # 19 values, linear
  learning_rate:
    fun: "np.linspace"
    args: [0.001, 0.1]
    kwargs: {num: 10, endpoint: true}  # 10 values, exact spacing
# Total: 20 × 19 × 10 = 3,800 combinations
```
