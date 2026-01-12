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

## Installation

### Prerequisites
- **Boost C++ libraries** (with shared libraries compiled)
- **Python 3.8+**
- **Python packages**: numpy, pyyaml
- (Optional) SLURM cluster access for distributed simulations

### Setup

```bash
# Clone the repository
git clone https://github.com/[username]/fastHDMF-code.git
cd fastHDMF-code

# 1. Compile the C++ DMF extension
cd dynamic_fic_dmf_Cpp
python setup.py install
cd ..
```

> **⚠️ Important**: Before running `python setup.py install`, ensure that `setup.py` points to the correct Boost shared library (`.so`) files on your system. Follow the detailed installation instructions from the [original fastDMF repository](https://gitlab.com/concog/fastdmf) for guidance on locating and linking Boost libraries.

```bash
# 2. Install the Python experiment management package
pip install -e .
```

---

## Usage

### Running Experiments

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


### Configuration

Experiments are defined via YAML files in `configs/`. Example structure:

```yaml
simulation:
  nb_steps: 50000
  G: 2.9
  with_plasticity: true
  
data:
  sc_root: "Averaged_SCs/aal"
  
output:
  observables:
    - name: fc
      signal: bold
```

See `configs/Default.yaml` for all available parameters.

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