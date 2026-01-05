from setuptools import setup, Extension

ext = Extension('_DYN_FIC_DMF',
                libraries = ['boost_python312', 'boost_numpy312'],
                sources   = ['fastdyn_fic_dmf/DYN_FIC_DMF.cpp'],
                include_dirs = ['/opt/homebrew/Cellar/boost/1.83.0/include','/opt/homebrew/Cellar/boost-python3/1.83.0_1/include',
                                ],  # Replace 1.xx.x_1 with your Boost version
                library_dirs = ['/opt/homebrew/Cellar/boost/1.83.0/lib','/opt/homebrew/Cellar/boost-python3/1.83.0_1/lib'],
                runtime_library_dirs=['/opt/homebrew/Cellar/boost/1.83.0/lib','/opt/homebrew/Cellar/boost-python3/1.83.0_1/lib'],
                extra_compile_args=['-std=c++11']
                )


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

