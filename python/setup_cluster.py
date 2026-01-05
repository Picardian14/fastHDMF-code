from setuptools import setup, Extension
import os 
eigen_path = os.path.abspath('cpp/Eigen')

ext = Extension('_DYN_FIC_DMF',
                libraries = ['boost_python38', 'boost_numpy38'],
                sources   = ['fastdyn_fic_dmf/DYN_FIC_DMF.cpp'])

setup(name              = 'fastdyn_fic_dmf',
      version          = '0.1',
      description      = 'Fast Dynamic Mean Field simulator of neural dynamics',
      author           = 'Pedro A.M. Mediano',
      author_email     = 'pam83@cam.ac.uk',
      url              = 'https://gitlab.com/concog/fastdmf',
      long_description = open('../README.md').read(),
      package_data     = {'python/fastdyn_fic_dmf': ['DTI_fiber_consensus_HCP.csv']},
      install_requires = ['numpy'],
      ext_modules      = [ext],
      packages         = ['fastdyn_fic_dmf'])

