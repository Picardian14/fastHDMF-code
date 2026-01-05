"""
Experiment management system for HDMF simulations
Handles configuration loading, logging, and result organization
"""
import yaml
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import re
import psutil

from fastHDMF.observables import ObservablesPipeline
from fastHDMF.utils.calculate_grid_size import get_grid_size
from fastHDMF.utils.data_loading import load_metadata, load_all_sc_matrices

class ExperimentManager:
    """Manages HDMF experiments with configuration, logging, and result storage"""
    
    def __init__(
        self,
        experiment_id: str,
        project_root: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        job_id: Optional[int] = None,
        job_count: Optional[int] = None,
        verbose: bool = False,
    ):
        # Auto-detect project root from package location if not provided
        if project_root is None:
            # Assume repo root is parent of the fastHDMF package directory
            package_dir = Path(__file__).parent
            project_root = package_dir.parent
        
        self.project_root = Path(project_root).resolve()
        self.configs_dir = self.project_root / "configs" 
        self.data_dir = self.project_root / "data"  # data directory
        # unified results directory
        self.results_dir = Path(results_dir) if results_dir else (self.project_root / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # By default, keep notebooks quiet (no console logs)
        self.verbose = bool(verbose)
        
        # placeholders for current experiment
        self.current_experiment = None
        self.experiment_dir = None
        self.logger = None
        self.current_config = None
        # store the experiment id (filename stem) user passed; we keep original arg name consistent externally
        self.experiment_id = experiment_id
        self.observables = None
        
        # if config_path provided, immediately set up experiment
        if experiment_id:
            self.setup_experiment(
                experiment_id,
                job_id=str(job_id) if job_id is not None else None,
                job_count_str=str(job_count) if job_count is not None else None
            )
        # Load patient data once
        self.metadata = load_metadata(
            datapath=self.data_dir,
            metadata_file=self.current_config.get('data', {}).get('metadata', None),
            sc_root=self.current_config.get('data', {}).get('sc_root', 'SCs')
        )
        self.all_ipps = self.metadata['IPP'].tolist() 
        self.sc_matrices = load_all_sc_matrices(
            ipp_list=self.all_ipps,
            datapath=self.data_dir,
            sc_root=self.current_config.get('data', {}).get('sc_root', 'SCs'),
            normalize=self.current_config.get('data', {}).get('normalize_sc', True),
            threshold=self.current_config.get('data', {}).get('threshold', 0),
            in_counts=self.current_config.get('data', {}).get('in_counts', False)
        ) if not self.metadata.empty else {}
        self.logger.info(f"Loaded SC matrices for {len(self.sc_matrices)} patients from {self.current_config.get('data', {}).get('sc_root', 'SCs')} with config threshold {self.current_config.get('data', {}).get('threshold', 0)}")
    
    def integrate_local_results(self) -> None:
        res = self.load_experiment_results()
        full_results = res['full_results']
        if 'grid' in self.current_config:
            _,grid_shape = get_grid_size(self.current_config_path)
            for k, data in full_results['observables'].items():
                element_shape = data.flatten()[0].shape            
                # combine grid dimensions with element dimensions
                full_shape = tuple(grid_shape) + element_shape
                full_results['observables'][k] = np.stack(data.flatten()).reshape(full_shape)
        else: 
            self.logger.info("No grid found - no grid reshaping")
        outp = self.experiment_dir / "full_results.pkl"
        with open(outp, 'wb') as f:
            pickle.dump(full_results, f)


    def integrate_slurm_results(self) -> Path:
        base_dir = self.experiment_dir
        if not base_dir.exists() or not base_dir.is_dir():
            raise ValueError(f"Base experiment directory not found: {base_dir}")

        job_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('job_')]
        if not job_folders:
            raise ValueError(f"No job folders found inside base experiment directory: {base_dir}")

        # Numeric sort ensures global task order
        job_folders.sort(key=lambda x: int(x.name.split('_')[-1]))

        # Logging
        self.experiment_dir = base_dir
        self._setup_logging(self.experiment_id)
        self.logger.info(f"Integrating {len(job_folders)} job folders from {base_dir}")

        merged = None
        all_failed = set()

        for jf in job_folders:
            rp = jf / "full_results.pkl"
            if not rp.exists():
                self.logger.warning(f"Missing results in {jf}; skipping.")
                continue
            with open(rp, 'rb') as f:
                jr = pickle.load(f)

            # Collect failures if present
            all_failed.update(jr.get('failed_simulations', []))

            if merged is None:
                # First job: take structure as-is
                merged = jr
                continue

            # Concatenate observables along task axis (axis 0)
            tgt_obs = merged.get('observables', {})
            src_obs = jr.get('observables', {})
            for k, src in src_obs.items():
                if k in tgt_obs and getattr(tgt_obs[k], 'size', 0) > 0 and getattr(src, 'size', 0) > 0:
                    tgt_obs[k] = np.concatenate([tgt_obs[k], src], axis=0)
                elif getattr(src, 'size', 0) > 0:
                    tgt_obs[k] = src                        
            merged['observables'] = tgt_obs

            # Update simple counters
            m = merged.setdefault('meta', {})
            m['num_tasks_integrated'] = m.get('num_tasks_integrated', 0) + jr.get('meta', {}).get('local_task_count', 0)
        
        if 'grid' in self.current_config:
            _, grid_shape = get_grid_size(self.current_config_path,verbose=False)
            # Determine element shape per observable and reshape into full grid shape        
            for k, data in merged['observables'].items():
                # extract element shape (all dims after the first axis)
                element_shape = data.flatten()[0].shape            
                # combine grid dimensions with element dimensions
                full_shape = tuple(grid_shape) + element_shape
                #merged['observables'][k] = np.stack(data.flatten()).reshape(full_shape)
                flat = data.flatten()
                sample = next((x for x in flat if x is not None), None)
                if sample is None:
                    merged['observables'][k] = np.array(list(flat), dtype=object).reshape(grid_shape)
                else:
                    element_shape = getattr(sample, 'shape', ())
                    full_shape = tuple(grid_shape) + element_shape
                    merged['observables'][k] = np.stack([x if x is not None else np.full(element_shape, np.nan) for x in flat]).reshape(full_shape)
        else: 
            self.logger.info("Using task or tasks_list mode - no grid reshaping")

        # store element shapes in metadata for reference
        
        # Finalize metadata
        merged = merged or {}
        merged.setdefault('meta', {})
        merged['meta'].update({
            'status': 'completed',
            'integrated_from_jobs': len(job_folders),
        })
        merged['failed_simulations'] = sorted(all_failed)

        # Save integrated
        outp = base_dir / "full_results.pkl"
        with open(outp, 'wb') as f:
            pickle.dump(merged, f)

        # Update / write metadata.json at base level
        meta_path = base_dir / "metadata.json"
        base_meta = {
            'experiment_id': self.experiment_id,
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'total_patients': len(merged.get('patients', [])),
            'integrated_from_jobs': len(job_folders),
            'original_job_folders': [d.name for d in job_folders],
            'failed_simulations': merged.get('failed_simulations', []),
            'observables_spec': self.observables.spec() if self.observables else [],
            'original_config_path': self.current_config_path,
            'config': self.current_config,
        }
        with open(meta_path, 'w') as f:
            json.dump(base_meta, f, indent=2)

        # Optionally archive job folders
        jobs_dir = base_dir / ".jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for jf in job_folders:
            try:
                shutil.move(str(jf), str(jobs_dir / jf.name))
            except Exception as e:
                self.logger.warning(f"Failed to move {jf.name}: {e}")

        self.logger.info(f"Integrated results saved to {outp}")
        return base_dir

    def load_config(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment configuration by experiment_id (filename without .yaml).
        Simple: search under configs/ (recursively) for a file named <experiment_id>.yaml.
        No over-engineering."""
        # ensure we have the final filename
        filename = experiment_id if experiment_id.endswith('.yaml') else f"{experiment_id}.yaml"
        self.logger_info(f"Searching for config file: {filename}")
        self.logger_info(f"Configs directory: {self.configs_dir}")
        # search all matches under configs
        matches = list(self.configs_dir.rglob(filename))
        self.logger_info(f"{experiment_id}")
        self.logger_info(f"Found config matches: {matches}")
        if not matches:
            # also allow absolute / relative direct path provided
            direct = Path(filename)
            if direct.exists():
                matches = [direct]
        if not matches:
            raise FileNotFoundError(f"Config file not found for experiment_id: {experiment_id}")
        # pick the first match deterministically (sorted)
        matches.sort()
        path = matches[0]
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        self.current_config_path = str(path)
        self.logger_info(f"Loaded config from: {path}")
        return config

    def setup_experiment(self, experiment_id: str, job_id: Optional[str] = None, job_count_str: Optional[str] = None) -> Tuple[Optional[Path], str]:
        """Setup a new experiment with logging and result directories.           
        """
        self.current_config = self.load_config(experiment_id)
        # Extract file stem as experiment id (user already passed it)
        self.experiment_id = Path(experiment_id).stem
        # Build and store the observables pipeline for this experiment
        try:
            self.observables = ObservablesPipeline.from_config(self.current_config.get("output", {}))
        except Exception:
            # Fallback to a safe default if config is malformed
            self.observables = ObservablesPipeline.default()
             

        # Determine if we will save anything; only then create directories
        out = (self.current_config or {}).get('output', {})

        will_save = bool(out.get('save_full_outputs') or out.get('save_metrics_only') or out.get('save_plots'))

        if out.get("results_dir"):
            self.results_dir = Path(self.results_dir) / out.get("results_dir") # Added to the assumed baseline path
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir = None
        if will_save:
            self.experiment_dir = Path(self.results_dir) / self.experiment_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # New directory layout for SLURM array jobs:        
        #   Each job has subfolder: job_<id>
        # Single (non-array) runs keep old flat structure.
        if job_id is not None and will_save:
            #base_experiment_id = f"{self.experiment_id}_{timestamp}"
            self.base_experiment_id = self.experiment_id  # informational
            self.job_id = int(job_id)
            self.job_count = int(job_count_str) if job_count_str is not None else None

            base_dir = Path(self.results_dir) / self.experiment_id
            base_dir.mkdir(parents=True, exist_ok=True)

            job_folder = base_dir / f"job_{job_id}"
            job_folder.mkdir(parents=True, exist_ok=True)
            self.experiment_dir = job_folder

            # Maintain full self.experiment_id with job suffix for metadata/logging
            self.experiment_id = f"{self.experiment_id}_job{job_id}"

        # Setup logging (console-only if not saving)
        self._setup_logging(self.experiment_id)

    
        metadata = {
            'experiment_id': self.experiment_id,
            'original_config_path': self.current_config_path or str(experiment_id),
            'start_time': datetime.now().isoformat(),
            'config': self.current_config,
            'status': 'initialized',
            'observables_spec': self.observables.spec() if self.observables else []
        }
        # Only persist metadata if saving
        if self.experiment_dir is not None:
            metadata_path = self.experiment_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.current_experiment = self.experiment_id
        self.logger.info(f"Experiment '{self.experiment_id}' initialized")
        desc = (self.current_config.get('experiment') or {}).get('description')
        if desc:
            self.logger.info(f"Config: {desc}")
        if self.experiment_dir is not None:
            self.logger.info(f"Results will be saved to: {self.experiment_dir}")
        else:
            self.logger.info("No save flags set; results will not be written to disk.")

        return self.experiment_dir, self.experiment_id
    
    def _setup_logging(self, experiment_id: str):
        """Setup logging for the experiment.

        - Always logs to file when experiment_dir is set.
        - Logs to console only when self.verbose is True.
        """

        # Create logger
        self.logger = logging.getLogger(f"hdmf_experiment_{experiment_id}")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate notebook output via root logger
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler (only when verbose)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Optional file handler if saving to a directory
        if self.experiment_dir is not None:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # If neither console nor file handlers were added, keep logger silent and safe.
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        
    def logger_info(self, message: str):
        """Log info message (creates basic logger if none exists)"""
        if self.logger:
            self.logger.info(message)
        else:
            if self.verbose:
                print(f"INFO: {message}")
    
    def save_results(self, results: Dict[str, Any], config: Dict[str, Any], failed_simulations: Optional[List] = None):
        """Save experiment results"""
        out = (config or {}).get('output', {})
        will_save = bool(out.get('save_full_outputs') or out.get('save_metrics_only') or out.get('save_plots'))
        if not will_save:
            # Nothing to persist; just log and return
            self.logger_info("Run completed (no save flags set); skipping filesystem writes.")
            return self.experiment_dir
        if not self.experiment_dir:
            # Create directory on-demand if missing (should not happen if setup_experiment followed flags)
            exp_id = (self.current_config_path and Path(self.current_config_path).stem) or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = Path(self.results_dir) / exp_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            # Ensure logging to file is enabled now
            self._setup_logging(exp_id)
        
        # Save full results if requested
        if config['output']['save_full_outputs']:
            results_path = self.experiment_dir / "full_results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Saved full results to: {results_path}")
        
        # Determine patient count from new structured results if present
        if isinstance(results, dict) and 'patients' in results:
            total_patients = len(results['patients'])
        else:
            # Fallback: try common patterns or length heuristic
            total_patients = len(results)

        # Update experiment metadata
        self._update_experiment_status('completed', {
            'total_patients': total_patients,
            'failed_simulations': failed_simulations or []
        })
        
        self.logger.info(f"Experiment completed successfully!")
        return self.experiment_dir
    
    def _update_experiment_status(self, status: str, additional_data: Dict[str, Any] = None):
        """Update experiment metadata"""
        if not self.experiment_dir:
            return
        metadata_path = self.experiment_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = status
        metadata['end_time'] = datetime.now().isoformat()
        
        if additional_data:
            metadata.update(additional_data)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_experiments(self) -> pd.DataFrame:
        """List all completed experiments"""
        experiments = []
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    experiments.append({
                        'experiment_id': metadata.get('experiment_id'),
                        'name': metadata.get('config', {}).get('experiment', {}).get('name'),
                        'description': metadata.get('config', {}).get('experiment', {}).get('description'),
                        'status': metadata.get('status'),
                        'start_time': metadata.get('start_time'),
                        'total_patients': metadata.get('total_patients'),
                        'directory': str(exp_dir)
                    })
        
        return pd.DataFrame(experiments)

    def load_experiment_results(self) -> Dict[str, Any]:
        """Load results from a completed experiment. As with setup_experiment it sets the current results directory"""        

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

        results = {}
        
        # Load metadata
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                results['metadata'] = json.load(f)
        # Load configuration from the original YAML file rather than from metadata JSON
           
        self.current_config = self.load_config(self.experiment_id)
        # Rebuild observables pipeline for any follow-up processing
        try:
            self.observables = ObservablesPipeline.from_config(self.current_config.get("output", {}))
        except Exception:
            self.observables = ObservablesPipeline.default()
        
        # Load full results if available
        full_results_path = self.experiment_dir / "full_results.pkl"
        if full_results_path.exists():
            with open(full_results_path, 'rb') as f:
                results['full_results'] = pickle.load(f)
        
        return results
