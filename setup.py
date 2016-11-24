from distutils.core import setup
from Cython.Build import cythonize


def main():
    setup(name='tropical',
          version='1.0',
          description='Dynamic analysis of systems biology models',
          author='Oscar Ortega',
          author_email='oscar.ortega@vanderbilt.edu',
          packages=['tropical'],
          requires=['pysb'],
          keywords=['systems', 'biology', 'model'],
          ext_modules=cythonize("tropical/choose_max.pyx")
          )

if __name__ == '__main__':
    main()
