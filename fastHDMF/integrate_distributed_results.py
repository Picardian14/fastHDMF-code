#!/usr/bin/env python3
"""
Command-line script to run HDMF experiments
Usage: python run_experiment.py <config_name> [experiment_id]
"""
import sys
import os
import argparse
from pathlib import Path

from fastHDMF.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description='Integrate results already ran as a SLURM job')
    parser.add_argument('id', help='Name of the experiment to be integrated')    
    
    args = parser.parse_args()
    print(args)    
    
    try:
        # manager auto-inits to results_dir under project_root, no config needed here
        experiment_manager = ExperimentManager(experiment_id=args.id, results_dir="/network/iss/cohen/data/Ivan/fastHDMF")
        _ = experiment_manager.integrate_slurm_results()
        print("\n✅Results got integrated")

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
