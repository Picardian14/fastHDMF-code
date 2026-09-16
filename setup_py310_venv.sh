#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
KERNEL_NAME="fasthdmf-py310"
KERNEL_DISPLAY_NAME="Python 3.10 (.venv fastHDMF)"

cd "${PROJECT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}. Install Python 3.10 first, or run:"
  echo "  PYTHON_BIN=/path/to/python3.10 ./setup_py310_venv.sh"
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install \
  numpy==1.23.5 \
  scipy \
  matplotlib \
  pyyaml \
  joblib \
  ipykernel \
  notebook

"${VENV_DIR}/bin/python" -m pip install -e .

mkdir -p \
  "${PROJECT_DIR}/.vscode" \
  "${VENV_DIR}/.jupyter" \
  "${VENV_DIR}/.ipython" \
  "${VENV_DIR}/.cache/matplotlib"

cat > "${PROJECT_DIR}/.env" <<EOF
JUPYTER_CONFIG_DIR=${VENV_DIR}/.jupyter
JUPYTER_DATA_DIR=${VENV_DIR}/share/jupyter
IPYTHONDIR=${VENV_DIR}/.ipython
MPLCONFIGDIR=${VENV_DIR}/.cache/matplotlib
EOF

cat > "${PROJECT_DIR}/.vscode/settings.json" <<'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.envFile": "${workspaceFolder}/.env",
  "python.terminal.activateEnvironment": true,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "terminal.integrated.env.linux": {
    "JUPYTER_CONFIG_DIR": "${workspaceFolder}/.venv/.jupyter",
    "JUPYTER_DATA_DIR": "${workspaceFolder}/.venv/share/jupyter",
    "IPYTHONDIR": "${workspaceFolder}/.venv/.ipython",
    "MPLCONFIGDIR": "${workspaceFolder}/.venv/.cache/matplotlib"
  }
}
EOF

JUPYTER_CONFIG_DIR="${VENV_DIR}/.jupyter" \
JUPYTER_DATA_DIR="${VENV_DIR}/share/jupyter" \
IPYTHONDIR="${VENV_DIR}/.ipython" \
MPLCONFIGDIR="${VENV_DIR}/.cache/matplotlib" \
  "${VENV_DIR}/bin/python" -m ipykernel install \
    --prefix "${VENV_DIR}" \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

MPLCONFIGDIR="${VENV_DIR}/.cache/matplotlib" \
  "${VENV_DIR}/bin/python" -c \
  "import sys, numpy, scipy, matplotlib, yaml, joblib, fastHDMF; print(sys.executable); print(sys.version.split()[0]); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('matplotlib', matplotlib.__version__); print('pyyaml', yaml.__version__); print('joblib', joblib.__version__); print('fastHDMF import ok')"

"${VENV_DIR}/bin/python" -m pip check

echo
echo "Environment ready at ${VENV_DIR}"
echo "VS Code interpreter: ${VENV_DIR}/bin/python"
echo "Jupyter kernel: ${KERNEL_DISPLAY_NAME}"
