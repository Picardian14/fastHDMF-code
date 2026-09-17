# fastHDMF: Homeostatic Dynamic Mean Field Model

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-b31b1b.svg)](https://doi.org/xxx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the code accompanying the article:

> **"[The impact of homeostatic inhibitory plasticity in a generative biophysical model]"**  
> Mindlin et al., Biorxiv, 2026.  

---

## Overview

This repository provides:

1. **Modified fastDMF Implementation** — A C++/MEX extension of the Dynamic Mean Field model by [Herzog et al.](https://doi.org/10.1162/netn_a_00410), incorporating homeostatic plasticity mechanisms (`dynamic_fic_dmf_Cpp/`).

2. **Experiment Management Toolkit** — A Python framework (`fastHDMF/`) for configuring, running, and analyzing large-scale simulations with SLURM cluster support.

3. **Reproducibility Resources** — Configuration files and notebooks to reproduce all figures from the manuscript (`configs/`, `notebooks/`).

> 📌 A pure-Python implementation of HDMF is also available at [carlosmig/Homo_DMF](https://github.com/carlosmig/Homo_DMF).

---

## Docker notebook environment

The Docker image contains a matched Python 3.10, NumPy, Boost.Python, and
Boost.NumPy environment, the compiled simulator, the `fastHDMF` package,
JupyterLab, and the dependencies used by the repository notebooks. The host
only needs Docker.

Start the notebook server in the background from the repository root:

```bash
docker compose up --build -d
```

The repository is mounted at `/work`, so notebook changes and generated files
are retained on the host. Open <http://127.0.0.1:8888/lab> in a browser. To
watch startup logs or stop the server, use:

```bash
docker compose logs -f notebook
docker compose down
```

The port is bound only to the host's loopback interface. Jupyter authentication
is therefore disabled for convenient local development; do not change the
port mapping to `0.0.0.0:8888:8888` on an untrusted network.

### Connect from VS Code

1. Install the Microsoft **Python** and **Jupyter** VS Code extensions.
2. Start the container with `docker compose up --build -d`.
3. Open an `.ipynb` file, choose **Select Kernel**, then **Existing Jupyter
   Server**, and enter `http://127.0.0.1:8888`.
4. Select the Python 3 kernel offered by that server.

You can also use VS Code's **Dev Containers: Attach to Running Container...**
command and select the Compose `notebook` container.

### Docker without Compose

The equivalent direct Docker commands are:

```bash
docker build -t fasthdmf-notebook .
docker run --name fasthdmf-notebook -d \
  --restart unless-stopped \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD:/work" \
  fasthdmf-notebook
```

To run the built-in simulator smoke test or open a shell in the running
container:

```bash
docker compose exec notebook python3 /opt/fastdyn_fic_dmf/smoke_test.py
docker compose exec notebook bash
```

---

## Usage

### Direct Python API

The core simulation is run via the `dmf.run()` function. Example:

```python
import fastdyn_fic_dmf as dmf
import numpy as np

# Load structural connectivity
C = np.loadtxt('data/SCs/Averaged_SCs/aal/healthy_average.csv', delimiter=',')
C = 0.2 * C / np.max(C)
N = C.shape[0]

# Set target rate
obj_rate = 3.44
# Set plasticity parameters
LR = 3.5 * np.ones(N)  # Learning rate per region
DECAY = 10000
 # Decay time constant per region
## To note, these parameters can be set with the homeostatic rules that relates both paramters with DECAY = np.exp(a + np.log(LR) * b) 
## The slope 'b' and and intercept 'a'  have to be found for the used connectivity matrix

# Configure simulation parameters
params = dmf.default_params(C=C, lrj=LR, taoj=DECAY)
params['obj_rate'] = obj_rate
params['with_decay'] = True      # Enable homeostatic decay
params['with_plasticity'] = True  # Enable synaptic plasticity
params['G'] = 3.5                 # Global coupling strength
params['J'] = 0.75 * params['G'] * params['C'].sum(axis=0) + 1

# Run simulation
rates, rates_inh, bold, fic = dmf.run(params, nb_steps=50000)
```

**Key parameters:**
- `with_decay`: Enable/disable homeostatic decay mechanism
- `with_plasticity`: Enable/disable synaptic plasticity
- `lrj`: Learning rate (scalar or per-region vector)
- `taoj`: Decay time constant (scalar or per-region vector)

See [examples.ipynb](notebooks/examples.ipynb) for detailed usage examples.

### Running Experiments with Configuration Files

The `fastHDMF.ExperimentManager` provides a YAML-based workflow for managing large-scale simulations and simplifying cluster job submissions. While the main simulation is performed by `dmf.run()`, this toolkit handles configuration, parallelization, and result aggregation.

**Local execution:**
```bash
python -m fastHDMF.run_experiment <experiment_id> --config experiments/<config_name>
```

**SLURM cluster submission:**
```bash
cd slurm
./submit_experiment_slurm_array.sh
```
This will show the available experiments to run and let you define main SBATCH directives.

**Configuration example** (`configs/Default.yaml`):

```yaml
simulation:
  nb_steps: 50000
  G: 2.9
  with_plasticity: true
  with_decay: true
  lrj: 3.5
  
data:
  sc_root: "Averaged_SCs/aal"
  
output:
  observables:
    - name: fc
      signal: bold
```

See [configs/Default.yaml](configs/Default.yaml) for all available parameters.

---

## Repository Structure

```
fastHDMF-code/
├── fastHDMF/               # Python experiment management package
├── dynamic_fic_dmf_Cpp/    # C++/MEX DMF implementation
├── configs/                # Experiment configurations
│   └── experiments/        # Paper-specific configs
├── notebooks/              # Analysis and figure generation
├── slurm/                  # Cluster submission scripts
└── data/                   # Input data (SC matrices, receptor maps)
```

---

## Reproducing Paper Figures

Jupyter notebooks in `notebooks/` reproduce all manuscript figures:

| Notebook | Description |
|----------|-------------|
| `PaperFigures.ipynb` | Main manuscript figures |
| `Chimera_Calculator.ipynb` | Chimera state analysis |
| `examples.ipynb` | Usage examples and tutorials |

---


---

## Acknowledgments

This work builds upon the [fastDMF](https://gitlab.com/concog/fastdmf) framework by Herzog et al.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
