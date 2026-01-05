"""
HDMF simulation runner with experiment management
"""
import sys
from pathlib import Path
import numpy as np
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product, islice
from typing import Any, Dict, Tuple, Optional, Union
from collections import defaultdict
import hashlib

# Import modules
from fastHDMF.experiment_manager import ExperimentManager
from fastHDMF.helper_functions import filter_bold
from fastHDMF.observables import ObservablesPipeline

# Import HDMF (assuming it's available in the environment)
try:
    import fastdyn_fic_dmf as dmf
except ImportError:
    print("Warning: fastdyn_fic_dmf not available. Make sure you're in the correct environment.")
    dmf = None

class HDMFSimulationRunner:
    """Runs HDMF simulations with experiment management"""

    def __init__(self, experiment_manager: ExperimentManager, observables: Optional[ObservablesPipeline] = None):        
        self.exp = experiment_manager
        # Prefer pipeline from ExperimentManager; allow explicit override
        if observables is not None:
            self.observables = observables
        elif getattr(self.exp, "observables", None) is not None:
            self.observables = self.exp.observables
        else:
            self.observables = ObservablesPipeline.default()
        self.exp.logger.info(f"Using observables: {self.observables}")
        self.exp.logger.info(f"current config: {self.exp.current_config}")
        # Items is the cartesian products between SC matrices to process and paremeter values that you can parallelize over
        self.items = self.define_items_over(self.exp.current_config)
        # Tasks is how the parameters for the model are constructed. If not grid is defined, it will just have the main parameters that will be configured for the simulation
        self.tasks = list(self.define_tasks_from_config(self.exp.current_config))
        
    @property
    def all_ipps(self):
        """Return list of all subject IDs"""
        return list(self.exp.sc_matrices.keys()) 

    def _generate_unique_seed(self, task: dict, ipp: str, item_idx: int) -> int:
        """
        Generate a unique seed for each simulation based on:
        - Patient ID (ipp)
        - Item index within the experiment
        - Task parameters
        - Optional base seed from config
        """
        # Pull seed config from simulation config if available
        sim_cfg = getattr(self.exp, 'current_config', {}).get('simulation', {}) if getattr(self, 'exp', None) else {}
        # Support both new and legacy ways of configuring the strategy
        seed_strategy = sim_cfg.get('seed_strategy', task.get('seed_strategy', 'unique_per_simulation'))
        # Alias/legacy normalization
        aliases = {
            'unique': 'unique_per_simulation',
            'per_sim': 'unique_per_simulation',
            'per_task': 'unique_per_task',
            'subject_seed': 'per_subject',
            'per_subject': 'per_subject',
            'same_for_all': 'same_for_all',
        }
        seed_strategy = aliases.get(seed_strategy, seed_strategy)

        # Base seed precedence: simulation.base_seed > task.seed > 42
        base_seed = int(sim_cfg.get('base_seed', task.get('seed', 42)))

        # Build a stable task signature excluding volatile keys
        volatile_keys = {
            'sc_matrix', 'seed', 'seed_strategy', 'param_key', 'param_value'
        }
        task_params = {k: v for k, v in task.items() if k not in volatile_keys}

        # Try to order keys consistently; use axis names if available
        axis_names = getattr(self, '_axis_names', []) or list(task_params.keys())
        ordered_items = []
        for k in axis_names:
            if k in task_params:
                ordered_items.append((k, task_params[k]))
        # Include any remaining keys deterministically
        remaining = sorted([(k, v) for k, v in task_params.items() if k not in dict(ordered_items)])
        ordered_items.extend(remaining)
        task_signature = str(ordered_items)

        # Compose unique string per strategy
        if seed_strategy == 'unique_per_simulation':
            unique_str = f"base:{base_seed}|sim:{ipp}:{item_idx}|task:{task_signature}"
        elif seed_strategy == 'unique_per_task':
            unique_str = f"base:{base_seed}|task:{task_signature}"
        elif seed_strategy == 'same_for_all':
            unique_str = f"base:{base_seed}|global"
        elif seed_strategy == 'per_subject':
            unique_str = f"base:{base_seed}|subject:{ipp}"
        else:
            # Fallback to unique per simulation
            unique_str = f"base:{base_seed}|sim:{ipp}:{item_idx}|task:{task_signature}"
        
        # Generate deterministic but unique seed using hash
        hash_obj = hashlib.md5(unique_str.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)  # Use first 8 hex chars as integer
        self.exp.logger.info(
            f"\nGenerated seed (strategy={seed_strategy}): {seed} from signature: {unique_str}\n"
        )
        # Ensure seed is within reasonable range for random number generators
        return seed % (2**31 - 1)
    def _generate_parameter_values(self, spec: dict) -> np.ndarray:
        """Generate parameter values from spec with optional custom function"""
        if "fun" in spec:
            fun_name = spec["fun"]
            args = spec.get("args", [])
            kwargs = spec.get("kwargs", {})
            
            try:
                func = eval(fun_name)
                values = func(*args, **kwargs)
                return np.array(values, dtype=float)
            except Exception as e:
                raise ValueError(f"Error calling function '{fun_name}': {e}")
        else:
            # Default: start, end, step
            values = np.arange(spec["start"], spec["end"], spec["step"], dtype=float)
            return np.round(values, 12)

    def _contiguous_block(self, total: int, parts: int, k: int) -> tuple[int, int]:
        base, rem = divmod(total, parts)
        if k < rem:
            start = k * (base + 1)
            end   = start + (base + 1)
        else:
            start = rem * (base + 1) + (k - rem) * base
            end   = start + base
        return start, end

    def define_tasks_from_config(self, config: dict):
        sim_defaults = config.get("simulation", {})
        
        # Check for tasks_list first
        tasks_list = config.get("tasks_list")
        if tasks_list:
            self.exp.logger.info(f"Using pre-defined tasks list with {len(tasks_list)} tasks")
            
            # Set up axis info for compatibility
            self._axis_names = ['task_index']
            self._axis_values = [np.arange(len(tasks_list))]
            self._global_total_combos = len(tasks_list)
            
            # Create iterator over task indices instead of combinations
            combo_iter = enumerate(tasks_list)
            
        else:
            # Grid-based approach
            grid = config.get("grid")
            if not grid:
                # Single task, no grid.
                self._axis_names = ['task']
                self._axis_values = [np.array([0.0])]
                self._global_total_combos = 1
                yield sim_defaults.copy()
                return

            axis_names = list(grid.keys())
            axis_values = [self._generate_parameter_values(spec) for spec in grid.values()]

            self._axis_names = axis_names
            self._axis_values = axis_values
            self._global_total_combos = int(np.prod([len(v) for v in axis_values]))
            
            # Create iterator over grid combinations with indices
            combo_iter = enumerate(product(*axis_values))

        # Common job slicing logic for both cases
        job_id = getattr(self.exp, 'job_id', None)
        job_count = getattr(self.exp, 'job_count', None)

        if job_id is not None and job_count is not None:
            start, end = self._contiguous_block(self._global_total_combos, job_count, job_id)
            combo_iter = islice(combo_iter, start, end)

        # Common task generation logic
        for idx, combo_data in combo_iter:
            task = sim_defaults.copy()
            
            if tasks_list:
                # combo_data is the task dict from tasks_list
                task.update(combo_data)
            else:
                # combo_data is the parameter combination from grid
                task.update({name: float(val) for name, val in zip(self._axis_names, combo_data)})
            
            yield task



    def prepare_hdmf_params(self, task):
        """Prepare HDMF parameters from config and SC matrix"""
               
        # Base parameters
        params = dmf.default_params(C=task['sc_matrix'])
        params['N'] = task['sc_matrix'].shape[0]
        if 'seed' in task:
            params['seed'] = int(task['seed'])
        if task is None:
            print( "No task provided, returning default params only.")
            return params
        # Configure from config file
        params['obj_rate'] = task['obj_rate']
        # NVC sigmoid option
        params['nvc_sigmoid'] = task.get('nvc_sigmoid', True)
        params['nvc_r0']         =  params['obj_rate']  # baseline firing-rate
        params['nvc_u50']         = 12.0   # half-saturation (Hz): compression starts in this range
        params['nvc_match_slope'] = False # if using sigmoid, match slope at obj_rate
        params['nvc_k']          = 0.20    # maximum vasodilatory signal gain                    
        
        
        params["with_decay"] = task['with_decay']
        params["with_plasticity"] = task['with_plasticity']

        if params["with_plasticity"]:
            if 'lrj' in task:
                LR = task['lrj']
            else:
                LR = 1.0  # Default learning rate for plasticity
                self.exp.logger.info(f"No learning rate 'lrj' specified in task; using default LR={LR}")
            if 'taoj' not in task:
                # Load homeostatic parameters
                fit_res_path = self.exp.data_dir / "LinearCoeffs" / f"{self.exp.current_config['data']['sc_root']}_fit_res_{str(params['obj_rate']).replace('.', '-')}.npy"
                if fit_res_path.is_file() is False:
                    self.exp.logger.error(f"Homeostatic fit results not found at: {fit_res_path}. Using inputted or default taoj value.")
                    DECAY = task.get('taoj', 1000)
                    self.exp.logger.info(f"Setting homeostatic DECAY={DECAY:.5f} for LR={LR:.5f}")
                    params['taoj'] = DECAY
                else:   
                    self.exp.logger.info(f"Loading homeostatic fit results from: {fit_res_path}")
                    fit_res = np.load(str(fit_res_path))
                    b = fit_res[0]
                    a = fit_res[1]
                    DECAY = np.exp(a + np.log(LR) * b) if task['with_decay'] else 0
                    self.exp.logger.info(f"Setting homeostatic DECAY={DECAY:.5f} for LR={LR:.5f}")
                    params['taoj'] = DECAY
            else:
                DECAY = task['taoj']

            # Makes decay and lr heterogenizable, as J is.
            if 'lr_vector' in task:
                LR_vector = np.array(task['lr_vector'])
                assert len(LR_vector) == params['N'], f"LR vector length {len(LR_vector)} does not match number of regions {params['N']}"
                params['lr_vector'] = LR_vector
            else:
                params['lr_vector'] = np.ones(params['N']) * LR
            if 'taoj' not in task:
                TAOJ_vector = np.exp(a + np.log(params['lr_vector']) * b) 
                params['taoj_vector'] = TAOJ_vector
            else:
                params['taoj_vector'] = np.ones(params['N']) * DECAY

            
        # Global coupling
        params['G'] = task['G']
        if 'alpha' in task:
            params['alpha'] = task['alpha']
            params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
        else:
            params['alpha'] = 0.75
            params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1

        params['TR'] = task['TR']
        params['flp'] = task.get('flp', 0.008)
        params['fhp'] = task.get('fhp', 0.09)
        params['burnout'] = task.get('burnout', 8)  # in volumes (TRs)

        if task.get('nb_steps', 50000) < 1000: # Assume that user inputted volumes instead of steps
            params['nb_steps'] = int(np.ceil(task.get('nb_steps') * params['TR'] / params['dtt'])) # Burnout is taken after
        else:
            params['nb_steps'] = task.get('nb_steps', 50000)

        # Neuromodulation
        if 'wgaine' in task:
            params['wgaine'] = task['wgaine']
            # By defualt, set inhibitory gain to match excitatory gain
            params['wgaini'] = task['wgaine']
        if 'wgaini' in task:
            params['wgaini'] = task['wgaini']
        if 'receptors' in task:
            RECEPTORS = np.squeeze(np.load(self.exp.data_dir / "receptor_maps" / task.get('receptors', 'AAL_5ht2a.npy'))[:params['N']])
            RECEPTORS = RECEPTORS/max(RECEPTORS)-min(RECEPTORS)
            RECEPTORS = RECEPTORS - max(RECEPTORS) + 1
            params["receptors"] = RECEPTORS
        if 'w' in task:
            params['w'] = task['w']
        if 'mixed' in task: # Only for mixed cases where you want to load a prespecified FIC vector
            self.exp.logger.info("Loading FIC vector for mixed stability task")
            G_value = f"{params['G']:.2f}".replace('.', '')
            if 'wgaine' in task:
                w_value = f"{params['wgaine']:.2f}".replace('.', '')
                fname = project_root / "data" / "dyn_fics"  / f"mean_fic_w_{w_value}.npy"
            else:
                fname = project_root / "data" / "dyn_fics"  / f"mean_fic_G_{G_value}.npy"            
            J = np.load(fname)
            params['J'] = J

        # Return settings
        # If bold in any observable signal         
        params["return_bold"] = self.observables.needs("bold")
        params["return_fic"] = self.observables.needs("fic")
        params["return_rate"] = self.observables.needs("rates")
        params["return_rate_inh"] = self.observables.needs("rates_inh")
        self.exp.logger.info(f"Prepared HDMF params: {params}")
        
        return params
    
    def define_items_over(self, config: dict):
        """
        Define items over to process with a given task. Can be or not parallelized.
        Generally, if no 'over' is specified, just each item would be one SC matrix (subject).
        If 'over' is specified, it should be a single parameter to iterate over (e.g. G, lrj, alpha, etc).
        In that case, each item is a combination of (subject, param_value) generating a Cartesian product.
        Returns a list of (ipp, item) tuples where item contains sc_matrix and parameter values.
        """
        sim = config.get("simulation", {})
        over_config = sim.get('over')        
        
        # Load SC matrices based on test mode
        if config['data']['test_mode']:
            max_subs = config['data'].get('max_subjects_test', 2)
            sc_matrices = dict(islice(self.exp.sc_matrices.items(), max_subs))                            
        else:
            sc_matrices = self.exp.sc_matrices
        
        items = []
        
        if over_config is None:
            # No 'over' specified, just iterate over patients
            for ipp, sc_matrix in sc_matrices.items():
                item = {'sc_matrix': sc_matrix}
                items.append((ipp, item))
        else:

            # Assume that you iterate over 1 parameter (x amount of SC_matrices)
            param_name = list(over_config.keys())[0]
            param_values = self._generate_parameter_values(over_config[param_name])

            # Cartesian product between patients and parameter values
            for ipp, sc_matrix in sc_matrices.items():
                for param_value in param_values:
                    item = {
                        'sc_matrix': sc_matrix,
                        'param_key': param_name, # Get the name of the parameter as it is passed to the model
                        'param_value': param_value

                    }
                    items.append((ipp, item))
        
        return items
        

    def run_one_simulation(self, task: dict) -> dict:
        """Run a single HDMF simulation with given SC matrix and task parameters"""
        params = self.prepare_hdmf_params(task)
        # Run the simulation
        rates, rates_inh, bold, fic_t = dmf.run(params, params['nb_steps'])
        outputs = {}
        # Minimal processing on outputs
        if params.get('return_rate', True):
            rates = rates[:, int(params['burnout'] * (params['TR'] / params['dtt'])):]            
            outputs['rates'] = rates
        if params.get('return_rate_inh', True):
            rates_inh = rates_inh[:, int(params['burnout'] * (params['TR'] / params['dtt'])):]            
            outputs['rates_inh'] = rates_inh
        if params.get('return_bold', True):
            bold = bold[:, params['burnout']:]
            bold = filter_bold(bold, flp=params['flp'], fhp=params['fhp'], tr=params['TR'])
            outputs['bold'] = bold
        if params.get('return_fic', True):
            fic_t = fic_t[:, int(params['burnout'] * (params['TR'] / params['dtt'])):]
            outputs['fic'] = fic_t
        # Once desired variables are ready to be observed, compute observables
        obs_dict = self.observables.compute(outputs, params, self.exp.current_config)
        return obs_dict

    def run_experiment(self):
        """
        Simple + in-place integration:
        - Precompute patients and tasks
        - For each task, run all patients (optionally in parallel)
        - Write each observable directly into its preallocated grid at [task_idx, patient_idx]
        - Save once
        """
        em = self.exp
        log = em.logger
        config = em.current_config
        sim = config.get("simulation", {})        
        parallel = bool(sim.get("parallel", False))

        n_items = len(self.items)
        log.info(f"Loaded {n_items} patients.")

        # --- Build tasks (Cartesian product + optional job slicing) ---

        local_task_count = len(self.tasks)
        log.info(f"Local task count: {local_task_count}")

        axis_names = getattr(self, "_axis_names", [])
        axis_values = getattr(self, "_axis_values", [])
        global_total = int(getattr(self, "_global_total_combos", local_task_count))

        # --- Observable grids: lazily create per key, fill in-place ---
        # shape per grid: (local_task_count, n_items), dtype=object
        observable_grids: Dict[str, np.ndarray] = {}
        failed_simulations = set()

        # --- Main loop: fill grids directly ---        
        def get_grid(obs_key: str) -> np.ndarray:
            """Create the grid on first use; then reuse (no temp rows)."""
            g = observable_grids.get(obs_key)
            if g is None:
                if sim.get('averaged', False):
                    # You can average over the 'over' items and keep per subject simulations
                    n_subjects = len(self.all_ipps)
                    g = np.empty((local_task_count, 1), dtype=object)
                    g[:] = None  # explicit init
                else:
                    g = np.empty((local_task_count, n_items), dtype=object)
                    g[:] = None  # explicit init
                observable_grids[obs_key] = g
            return g

        # Determine parallel job cap: use user-specified override if provided, else default 32
        max_jobs = getattr(self.exp, 'max_cpus', None)
        cap = max_jobs if (isinstance(max_jobs, int) and max_jobs > 0) else 32
        n_jobs = min(len(self.items), min(cap, os.cpu_count() or 1))
        # --- Helpers ---
        def _run_one(current_task: Dict[str, Any], i: int, ipp: str, item_value: Any) -> Tuple[int, Optional[Dict[str, Any]]]:
            try:
                # If there is a param to iterate over, set it here
                thread_task = current_task.copy()  # capture current task from outer loop
                thread_task['sc_matrix'] = item_value['sc_matrix']
                if item_value.get('param_key') is not None:
                    thread_task[item_value['param_key']] = item_value['param_value']
                # If there a seed defined make sure it is always different for each simulation
            
                unique_seed = self._generate_unique_seed(thread_task, ipp, i)
                #self.exp.logger.info(f"Generated unique seed {unique_seed} for ipp {ipp}, item {i}, task {thread_task}")
                thread_task['seed'] = unique_seed
                # Run simulation
                out = self.run_one_simulation(
                    task=thread_task,   # captured from loop below                    
                )
                # expected: dict {obs_key: value}
                return i, out
            except Exception as e:
                log.error(f"[{ipp}] simulation failed: {e}")
                return i, None


        for t_idx, current_task in enumerate(self.tasks):
            log.info(f"[Task {t_idx+1}/{local_task_count}] {current_task}")

            # Run all patients for this task
            if parallel and n_jobs > 1:
                pairs = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_run_one)(current_task, i, ipp, item) for i, (ipp, item) in enumerate(self.items)
                )
            else:
                pairs = [_run_one(current_task, i, ipp, item) for i, (ipp, item) in enumerate(self.items)]
            
            # write results directly into grids, with optional averaging
            if sim.get('averaged', False):
                # number of unique ipps (subjects) per averaged‐slot
                n_subjects = len(self.all_ipps)
                avg_n_items = len(self.items) // n_subjects
                for item_idx, out in pairs:
                    if out is None:
                        continue
                    # map item_idx → averaged‐slot index
                    avg_idx = item_idx // avg_n_items
                    for obs_key, value in out.items():
                        grid = get_grid(obs_key)
                        # accumulate fractionally to build the mean
                        if grid[t_idx, avg_idx] is None:
                            grid[t_idx, avg_idx] = value / avg_n_items
                        else:
                            grid[t_idx, avg_idx] += value / avg_n_items
            else:
                for item_idx, out in pairs:
                    if out is None:
                        ipp = self.items[item_idx][0]
                        failed_simulations.add(ipp)
                        continue
                    for obs_key, value in out.items():
                        grid = get_grid(obs_key)
                        grid[t_idx, item_idx] = value

        # --- Package + save once ---
        axis_values_dict = {name: np.array(vals) for name, vals in zip(axis_names, axis_values)}
        meta = {
            "job_id": getattr(em, "job_id", None),
            "job_count": getattr(em, "job_count", None),
            "local_task_count": local_task_count,
            "global_total_combos": global_total,
            "partition_strategy": "contiguous_by_job_id",
            "observables_spec": list(observable_grids.keys()),
            "items": self.items,
        }

        results = {
            "param_axes": axis_names,
            "axis_values": axis_values_dict,            
            "observables": observable_grids,  # dict[str, np.ndarray] shape (tasks, items)
            "failed_simulations": sorted(failed_simulations),
            "meta": meta,
        }

        em.save_results(results=results, config=config, failed_simulations=results["failed_simulations"])
        log.info("Experiment finished and results saved.")
        return results