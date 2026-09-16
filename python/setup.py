from pathlib import Path
import sys

from setuptools import Extension, setup


HERE = Path(__file__).resolve().parent
PYTHON_SUFFIX = f"{sys.version_info.major}{sys.version_info.minor}"


extension = Extension(
    "_DYN_FIC_DMF",
    sources=["fastdyn_fic_dmf/DYN_FIC_DMF.cpp"],
    include_dirs=["/usr/include/eigen3"],
    libraries=[
        f"boost_python{PYTHON_SUFFIX}",
        f"boost_numpy{PYTHON_SUFFIX}",
    ],
    extra_compile_args=["-O3", "-g0", "-std=c++14", "-pthread"],
    extra_link_args=["-pthread"],
)


setup(
    name="fastdyn_fic_dmf",
    version="0.1.0",
    description="Fast Dynamic Mean Field simulator of neural dynamics",
    author="Pedro A.M. Mediano",
    author_email="pam83@cam.ac.uk",
    url="https://gitlab.com/concog/fastdmf",
    long_description=(HERE.parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["fastdyn_fic_dmf"],
    package_data={"fastdyn_fic_dmf": ["DTI_fiber_consensus_HCP.csv"]},
    install_requires=["numpy==1.23.5"],
    python_requires=">=3.10,<3.11",
    ext_modules=[extension],
)
