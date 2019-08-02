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
    from Cython.Distutils.build_ext import build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext
    use_cython = False
else:
    use_cython = True


# factory function
def my_build_ext(pars):
    # import delayed:

    # include_dirs adjusted:
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    # object returned:
    return build_ext(pars)

#### libs
if platform.system() == "Windows":
    math_lib = []
else:
    math_lib = ['m']

#### Python include
py_inc = [get_python_inc()]

#### cmdclass
cmdclass = {'build_py': build_py}

#### Extension modules
ext_modules = []
if use_cython:
    cmdclass.update({'build_ext': my_build_ext})
    ext_modules += [Extension("pydyno.lcs",
                              ["pydyno/lcs/clcs.c",
                               "pydyno/lcs/lcs.pyx"],
                              libraries=math_lib,
                              include_dirs=py_inc)]
else:
    cmdclass.update({'build_ext': my_build_ext})
    ext_modules += [Extension("pydyno.lcs",
                              ["pydyno/lcs/clcs.c",
                               "pydyno/lcs/lcs.c"],
                              libraries=math_lib,
                              include_dirs=py_inc)]

install_requires = ['pysb', 'matplotlib', 'anytree', 'scikit-learn', 'pydot',
                    'editdistance', 'pandas', 'networkx']


setup(name='pydyno',
      version='0.1.1',
      description='Dynamic analysis of systems biology models',
      author='Oscar Ortega',
      author_email='oscar.ortega@vanderbilt.edu',
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires,
      setup_requires=build_requires,
      test_suite='nose.collector',
      tests_require=['pathos', 'hdbscan'],
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      keywords=['systems', 'biology', 'model'],
      )
