# from distutils.core import setup
from setuptools import setup, find_packages


setup(name='DynSign',
      version='1.0',
      description='Dynamic analysis of systems biology models',
      author='Oscar Ortega',
      author_email='oscar.ortega@vanderbilt.edu',
      packages=find_packages(),
      include_package_data=True,
      # packages=['DynSign', 'DynSign.visualization', 'DynSign.visualization.cytoscapejs'],
    requires=['pysb', 'pathos', 'sympy'],
      keywords=['systems', 'biology', 'model'],
      )

