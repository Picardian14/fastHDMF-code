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
        libgomp1 \
        python3 \
        python3-pip \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

RUN python3 -m pip install --no-cache-dir --no-index \
        --find-links=/wheels \
        numpy==1.23.5 \
        fastdyn_fic_dmf==0.1.0 \
    && rm -rf /wheels

# Notebook/runtime dependencies used by fastHDMF and the notebooks shipped in
# this repository. Versions are kept compatible with the NumPy ABI used to
# compile fastdyn_fic_dmf above.
RUN python3 -m pip install --no-cache-dir \
        bctpy==0.6.1 \
        httpx==0.27.2 \
        ipykernel==6.29.5 \
        joblib==1.4.2 \
        jupyterlab==4.2.5 \
        matplotlib==3.7.5 \
        nibabel==5.2.1 \
        nilearn==0.10.4 \
        notebook==7.2.2 \
        pandas==1.5.3 \
        psutil==6.1.1 \
        pyyaml==6.0.2 \
        scikit-learn==1.3.2 \
        scipy==1.10.1 \
        seaborn==0.13.2 \
        tqdm==4.67.1 \
    && python3 -m pip check

COPY docker/smoke_test.py /opt/fastdyn_fic_dmf/smoke_test.py

# Keep an image-local copy so `import fastHDMF` also works without a bind
# mount. When the repository is mounted at /work, that live copy takes
# precedence and notebook edits are immediately visible.
COPY fastHDMF /opt/fastHDMF/fastHDMF
COPY configs /opt/fastHDMF/configs

# Fail the image build if the compiled extension cannot run a simulation.
RUN python3 /opt/fastdyn_fic_dmf/smoke_test.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work:/opt/fastHDMF
WORKDIR /work

EXPOSE 8888

# The published port should be bound to 127.0.0.1 on the host. Authentication
# is disabled deliberately for a local development container.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/work", "--IdentityProvider.token="]
