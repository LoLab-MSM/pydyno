# from distutils.core import setup
from setuptools import setup, find_packages
from setuptools.extension import Extension
import sys
import platform
from distutils.sysconfig import get_python_inc
from setuptools.command.build_py import build_py

try:
    import numpy
except ImportError:  # We do not have numpy installed
    build_requires = ['numpy>=1.14']
else:
    # If we're building a wheel, assume there already exist numpy wheels
    # for this platform, so it is safe to add numpy to build requirements.
    # See gh-5184.
    build_requires = (['numpy>=1.14'] if 'bdist_wheel' in sys.argv[1:]
                      else [])

try:
    from Cython.Distutils.build_ext import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

#### libs
if platform.system() == "Windows":
    math_lib = []
else:
    math_lib = ['m']

#### Python include
py_inc = [get_python_inc()]

#### NumPy include
np_inc = [numpy.get_include()]

#### cmdclass
cmdclass = {'build_py': build_py}

#### Extension modules
ext_modules = []
if use_cython:
    cmdclass.update({'build_ext': build_ext})
    ext_modules += [Extension("tropical.lcs",
                              ["tropical/lcs/clcs.c",
                               "tropical/lcs/lcs.pyx"],
                              libraries=math_lib,
                              include_dirs=py_inc + np_inc)]
else:
    ext_modules += [Extension("tropical.lcs",
                              ["tropical/lcs/clcs.c",
                               "tropical/lcs/lcs.c"],
                              libraries=math_lib,
                              include_dirs=py_inc + np_inc)]

install_requires = ['pysb', 'sympy', 'numpy', 'networkx', 'seaborn', 'hdbscan',
                    'scikit-learn', 'editdistance']

setup(name='DynSign',
      version='1.0',
      description='Dynamic analysis of systems biology models',
      author='Oscar Ortega',
      author_email='oscar.ortega@vanderbilt.edu',
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires,
      setup_requires=build_requires,
      test_suite='nose.collector',
      tests_require=['pathos', 'pygraphviz'],
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      keywords=['systems', 'biology', 'model'],
      )
