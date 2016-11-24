from distutils.core import setup
from Cython.Build import cythonize


def main():
    setup(name='TroPy',
          version='1.0',
          description='Dynamic analysis of systems biology models',
          author='Oscar Ortega',
          author_email='oscar.ortega@vanderbilt.edu',
          packages=['tropy'],
          requires=['pysb'],
          keywords=['systems', 'biology', 'model'],
          ext_modules=cythonize("choose_max.pyx")
          )

if __name__ == '__main__':
    main()
