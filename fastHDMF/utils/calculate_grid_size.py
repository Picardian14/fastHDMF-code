#!/usr/bin/env python3
"""
Utility to calculate grid size and suggest optimal SLURM array configuration
"""
import sys
import yaml
import numpy as np
from pathlib import Path
DATAPATH = Path(__file__).parent.parent.parent / "data" / "SCs"

def get_grid_size(config_path, verbose=True):
    """Calculate total grid size from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if there's a tasks_list instead of grid
    tasks_list = config.get("tasks_list", [])
    if tasks_list:
        # If tasks_list exists, use its length as the base grid size
        total_combinations = len(tasks_list)
        if verbose:
            print(f"Tasks list found with {total_combinations} parameter combinations")
            # Show first few tasks for reference
            for i, task in enumerate(tasks_list[:3]):
                print(f"  Task {i+1}: {task}")
            if len(tasks_list) > 3:
                print(f"  ... and {len(tasks_list) - 3} more tasks")
        
        # For tasks_list, we don't have traditional grid values
        all_values = [[i for i in range(len(tasks_list))]]  # Just task indices
    else:
        # Original grid logic
        grid = config.get("grid", {})
        if not grid:
            print("No grid parameters found in config")
            return 1, [1]
        
        total_combinations = 1
        all_values = []
        for param_name, spec in grid.items():
            if "fun" in spec:
                # Custom function - need to evaluate to get size
                fun_name = spec["fun"]
                args = spec.get("args", [])
                kwargs = spec.get("kwargs", {})
                
                try:
                    func = eval(fun_name)
                    values = func(*args, **kwargs)
                    param_size = len(values)
                except Exception as e:
                    print(f"Error evaluating {fun_name}: {e}")
                    return None
            else:
                # Traditional start/end/step
                start = spec["start"]
                end = spec["end"] 
                step = spec["step"]
                values = np.arange(start, end, step)
                param_size = len(values)
            
            total_combinations *= param_size
            if verbose:
                print(f"{param_name}: {param_size} values")
            # print actual values for sanity check
            try:
                val_list = values.tolist()
            except Exception:
                val_list = list(values)
            if verbose:
                print(f"  values: {val_list}")
            all_values.append(val_list)
    
    # Handle optional "over" parameters (e.g., seeds, SC matrices selection)
    sim = config.get("simulation", {})
    over = sim.get("over", {}) or config.get("over", {})
    if over:
        if verbose:
            print("\nOver parameters:")
        total_over = 1
        for param_name, spec in over.items():
            if "fun" in spec:
                fun_name = spec["fun"]
                args = spec.get("args", [])
                kwargs = spec.get("kwargs", {})
                try:
                    func = eval(fun_name)
                    values = func(*args, **kwargs)
                except Exception as e:                    
                    print(f"Error evaluating {fun_name}: {e}")
                    return None
            else:
                start = spec["start"]
                end = spec["end"]
                step = spec["step"]
                values = np.arange(start, end, step)
            over_size = len(values)
            total_over *= over_size
            if verbose:
                print(f"{param_name}: {over_size} values")
            try:
                over_list = values.tolist()
            except Exception:
                over_list = list(values)
            if verbose:
                print(f"  values: {over_list}")
        if verbose:
            print(f"\nTotal over combinations: {total_over}")
        
        # Count SC matrices in specified sc_root
        data_cfg = config.get("data", {})
        sc_root = data_cfg.get("sc_root", "SCs")
        sc_dir = Path(DATAPATH / sc_root)
        if sc_dir.exists() and sc_dir.is_dir():
            sc_files = list(sc_dir.glob('*.csv'))
            sc_count = len(sc_files) if not data_cfg.get("test_mode") else data_cfg.get("max_subjects_test")
        else:
            sc_count = 0
        if verbose:
            print(f"SC matrices found in '{sc_root}': {sc_count}")
        
        # Compute items per task (SC matrices × over values)
        if config.get("simulation", {}).get("averaged", False):
            # If averaged, each SC matrix is a separate task
            items_per_task = sc_count
        else:
            items_per_task = total_over * sc_count
        if verbose:
            if tasks_list:
                print(f"Items per task (over × SC matrices): {items_per_task}")
                print(f"Total simulations: {total_combinations * items_per_task} (tasks × over × SC matrices)")
            else:
                print(f"Items per task (over × SC matrices): {items_per_task}")
    else:
        # No over parameters: count SC matrices only
        data_cfg = config.get("data", {})
        sc_root = data_cfg.get("sc_root", "SCs")
        sc_dir = Path(DATAPATH / sc_root)
        if sc_dir.exists() and sc_dir.is_dir():
            sc_files = list(sc_dir.glob('*.csv')) 
            sc_count = len(sc_files) if not data_cfg.get("test_mode") else data_cfg.get("max_subjects_test")
        else:
            sc_count = 0
        if verbose:
            print(f"SC matrices found in '{sc_root}': {sc_count}")
        items_per_task = sc_count
        if verbose:
            if tasks_list:
                print(f"Items per task (SC matrices): {items_per_task}")
                print(f"Total simulations: {total_combinations * items_per_task} (tasks × SC matrices)")
            else:
                print(f"Items per task (SC matrices): {items_per_task}")
    
    grid_shape = [len(v) for v in all_values] + [items_per_task]
    return total_combinations, grid_shape

def suggest_array_size(grid_size, max_jobs=None):
    """Suggest optimal array size based on grid size"""
    if max_jobs is None:
        # Default suggestions based on grid size
        if grid_size <= 25:
            return min(grid_size, 5)
        elif grid_size <= 100:
            return min(grid_size, 10)
        elif grid_size <= 500:
            return min(grid_size, 25)
        elif grid_size <= 2000:
            return min(grid_size, 50)
        else:
            return min(grid_size, 100)
    else:
        return min(grid_size, max_jobs)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_grid_size.py <config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    grid_size, grid_shape = get_grid_size(config_path)

    if grid_size is None:
        print("Error calculating grid size")
        sys.exit(1)
    
    print(f"\nTotal grid combinations: {grid_size}")
    
    suggested = suggest_array_size(grid_size)
    print(f"Suggested SLURM array size: {suggested}")
    print(f"Tasks per job: {grid_size // suggested} (remainder: {grid_size % suggested})")
    