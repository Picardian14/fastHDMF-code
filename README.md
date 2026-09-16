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

## Docker installation (core simulator)

The Docker image contains a matched Python 3.10, NumPy, Boost.Python, and
Boost.NumPy environment. The host only needs Docker; it does not need Python,
NumPy, Boost, or a compiler.

Build the image from the repository root:

```bash
docker build -t fastdyn-fic-dmf .
```

Run the built-in two-node simulation explicitly if you want to verify the
installed extension:

```bash
docker run --rm fastdyn-fic-dmf \
  python3 /opt/fastdyn_fic_dmf/smoke_test.py
```

To run your own script, mount its directory at `/work`. Because the image's
entrypoint is Python, pass the script path and its arguments directly:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  fastdyn-fic-dmf python3 simulation.py
```

The optional `--user` makes files created in `/work` belong to the current
host user. To open an interactive Python prompt instead, run:

```bash
docker run --rm -it fastdyn-fic-dmf python3 -i
```

### Jupyter notebooks

Jupyter Notebook and JupyterLab are installed in the image. The default
container command starts the token-protected Jupyter Notebook server, with the
repository mounted as its working directory.

The simplest way to build and start it is:

```bash
docker compose up --build
```

You can do the same without Compose:

```bash
docker build -t fastdyn-fic-dmf .
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD:/work" \
  fastdyn-fic-dmf
```

The server prints a URL containing an authentication token, such as
`http://127.0.0.1:8888/tree?token=...`. Open that URL in a browser.

To use the server from VS Code:

1. Install the Microsoft Python and Jupyter extensions.
2. Open an `.ipynb` file and select the kernel picker in the upper-right.
3. Select **Existing Jupyter Server** and paste the complete URL printed by
   the container, including `?token=...`.

Code cells then execute inside the container, where `fastdyn_fic_dmf` is
already installed. Notebooks and generated files remain in the repository on
the host because it is mounted at `/work`.

The `fastHDMF` experiment and SLURM management package is not included.

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
