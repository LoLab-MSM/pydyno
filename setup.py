from distutils.core import setup


def main():
    setup(name='tropical',
          version='1.0',
          description='Dynamic analysis of systems biology models',
          author='Oscar Ortega',
          author_email='oscar.ortega@vanderbilt.edu',
          packages=['tropical'],
          requires=['pysb'],
          keywords=['systems', 'biology', 'model'],
          )

if __name__ == '__main__':
    main()
