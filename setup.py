# from distutils.core import setup
from setuptools import setup, find_packages


setup(name='DynSign',
      version='1.0',
      description='Dynamic analysis of systems biology models',
      author='Oscar Ortega',
      author_email='oscar.ortega@vanderbilt.edu',
      packages=find_packages(),
      include_package_data=True,
      install_requires=['pysb', 'sympy', 'numpy', 'networkx', 'seaborn'],
      setup_requires=['nose'],
      tests_require=['pathos', 'pygraphviz'],
      keywords=['systems', 'biology', 'model'],
      )

