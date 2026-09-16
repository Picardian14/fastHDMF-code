#!/usr/bin/env bash
# install.sh — set up a Python venv with the fastdyn_fic_dmf C++ extension and fastHDMF
#
# Usage:
#   ./install.sh              # creates venv at ./venv
#   ./install.sh /path/venv   # creates venv at the given path
#
# Requirements (install once on the host):
#   Ubuntu/Debian:  sudo apt install build-essential libboost-python-dev libboost-numpy-dev
#   Fedora/RHEL:    sudo dnf install gcc-c++ boost-python3-devel
#   macOS:          brew install boost-python3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-${REPO_ROOT}/venv}"

info() { echo "[install] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ── Boost detection (before picking Python, so we can match versions) ─────────

info "Looking for Boost.Python/Boost.Numpy ..."

BOOST_LIB_DIR=""
BOOST_SUFFIX=""
BOOST_SEARCH_DIRS=(/usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib
                   /opt/homebrew/lib /opt/homebrew/opt/boost/lib)

for dir in "${BOOST_SEARCH_DIRS[@]}"; do
    # Try any versioned libboost_python3XY.so, then unversioned
    for candidate in "${dir}"/libboost_python3*.so "${dir}"/libboost_python3*.dylib \
                     "${dir}"/libboost_python.so "${dir}"/libboost_python.dylib; do
        [ -f "$candidate" ] || continue
        base=$(basename "$candidate")
        # Extract suffix: libboost_python314.so → 314, libboost_python.so → ""
        suffix=$(echo "$base" | sed 's/libboost_python\(.*\)\.\(so\|dylib\|a\)/\1/')
        BOOST_LIB_DIR="$dir"
        BOOST_SUFFIX="$suffix"
        break 2
    done
done
[ -n "$BOOST_LIB_DIR" ] || die "Boost.Python not found. Install: sudo apt install libboost-python-dev libboost-numpy-dev"

BOOST_INC_DIR=""
for dir in /usr/include /usr/local/include /opt/homebrew/include /opt/homebrew/opt/boost/include; do
    if [ -f "${dir}/boost/python.hpp" ]; then
        BOOST_INC_DIR="$dir"
        break
    fi
done
[ -n "$BOOST_INC_DIR" ] || die "Boost headers not found. Install: sudo apt install libboost-dev"

info "Boost libs: $BOOST_LIB_DIR (suffix='$BOOST_SUFFIX'), headers: $BOOST_INC_DIR"

# ── Python — must match Boost.Python version ──────────────────────────────────

# If BOOST_SUFFIX is a version number (e.g. "314"), find the matching python3.14
PYTHON=""
if [[ "$BOOST_SUFFIX" =~ ^3([0-9]+)$ ]]; then
    MINOR="${BASH_REMATCH[1]}"
    for candidate in "python3.${MINOR}" python3 python; do
        if command -v "$candidate" &>/dev/null; then
            ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
            if [ "$ver" = "$BOOST_SUFFIX" ]; then
                PYTHON=$(command -v "$candidate")
                break
            fi
        fi
    done
    [ -n "$PYTHON" ] || die "Boost is built for Python 3.${MINOR}, but python3.${MINOR} was not found."
else
    PYTHON=$(command -v python3) || die "python3 not found."
fi

info "Using $("$PYTHON" --version) at $PYTHON"

# ── Virtual environment ───────────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    info "Venv already exists at $VENV_DIR, skipping creation."
fi

PY="$VENV_DIR/bin/python"

# Bootstrap pip — necessary on Python 3.12+ Debian/Ubuntu venvs
info "Bootstrapping pip ..."
"$PY" -m ensurepip --upgrade 2>/dev/null || true

# ── Python dependencies ───────────────────────────────────────────────────────

info "Installing Python dependencies ..."
"$PY" -m pip install --quiet --upgrade pip setuptools wheel
"$PY" -m pip install --quiet \
    "numpy>=1.24" "scipy>=1.10" "pandas>=2.0" "pyyaml>=6.0" \
    "joblib>=1.3" "tqdm>=4.65" "psutil>=5.9" "matplotlib>=3.7"

# ── C++ extension ─────────────────────────────────────────────────────────────

info "Building C++ extension (fastdyn_fic_dmf) ..."

CPP_DIR="${REPO_ROOT}/python"
EIGEN_PATH="${REPO_ROOT}/dynamic_fic_dmf_Cpp"
SETUP_TMP="${CPP_DIR}/_setup_install.py"

[ -d "$CPP_DIR" ]    || die "Not found: $CPP_DIR"
[ -d "$EIGEN_PATH" ] || die "Not found: $EIGEN_PATH"

# Write a temporary setup.py with the detected paths hard-wired
cat > "$SETUP_TMP" << PYEOF
from setuptools import setup, Extension
import sysconfig

ext = Extension(
    '_DYN_FIC_DMF',
    sources=['fastdyn_fic_dmf/DYN_FIC_DMF.cpp'],
    include_dirs=['${EIGEN_PATH}', '${BOOST_INC_DIR}', sysconfig.get_path('include')],
    library_dirs=['${BOOST_LIB_DIR}'],
    runtime_library_dirs=['${BOOST_LIB_DIR}'],
    libraries=['boost_python${BOOST_SUFFIX}', 'boost_numpy${BOOST_SUFFIX}'],
    extra_compile_args=['-O3', '-std=c++14'],
)

setup(
    name='fastdyn_fic_dmf',
    version='0.1',
    packages=['fastdyn_fic_dmf'],
    package_data={'fastdyn_fic_dmf': ['DTI_fiber_consensus_HCP.csv']},
    install_requires=['numpy'],
    ext_modules=[ext],
)
PYEOF

(cd "$CPP_DIR" && "$PY" "$SETUP_TMP" install 2>&1)
rm -f "$SETUP_TMP"

info "C++ extension built and installed."

# ── fastHDMF ──────────────────────────────────────────────────────────────────

info "Installing fastHDMF ..."
"$PY" -m pip install --quiet -e "${REPO_ROOT}"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "Installation complete!"
echo "  Activate with:  source ${VENV_DIR}/bin/activate"
echo ""
