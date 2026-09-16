# Ubuntu 22.04 provides a matched Python 3.10 / Boost.Python 3.10 pair.
FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        g++ \
        libboost-numpy-dev \
        libboost-python-dev \
        libeigen3-dev \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY README.md ./
COPY dynamic_fic_dmf_Cpp/dyn_fic_DMF.hpp dynamic_fic_dmf_Cpp/dyn_fic_DMF.hpp
COPY python/setup.py python/setup.py
COPY python/fastdyn_fic_dmf python/fastdyn_fic_dmf

# Pin build tooling as well as NumPy so rebuilding the image does not silently
# change the extension's Python environment.
RUN python3 -m pip install --no-cache-dir \
        numpy==1.23.5 \
        setuptools==68.2.2 \
        wheel==0.41.2 \
    && python3 -m pip wheel --no-cache-dir --no-deps \
        --wheel-dir /wheels numpy==1.23.5 \
    && cd /src/python \
    && python3 setup.py bdist_wheel --dist-dir /wheels


FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libboost-numpy1.74.0 \
        libboost-python1.74.0 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

RUN python3 -m pip install --no-cache-dir --no-index \
        --find-links=/wheels \
        numpy==1.23.5 \
        fastdyn_fic_dmf==0.1.0 \
    && rm -rf /wheels

# Include the Jupyter Notebook server and JupyterLab UI in the same image while
# keeping the simulator's Python and NumPy versions fixed.
RUN python3 -m pip install --no-cache-dir \
        jupyterlab==4.6.3 \
        notebook==7.6.2

COPY docker/smoke_test.py /opt/fastdyn_fic_dmf/smoke_test.py

# Fail the image build if the compiled extension cannot run a simulation.
RUN python3 /opt/fastdyn_fic_dmf/smoke_test.py

ENV HOME=/tmp \
    PYTHONUNBUFFERED=1
WORKDIR /work
EXPOSE 8888

# Starting the container starts a token-protected notebook server. A different
# command, such as `python3 simulation.py`, can still be supplied to docker run.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/work"]
