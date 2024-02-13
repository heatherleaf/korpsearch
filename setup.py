from setuptools import setup
from Cython.Build import cythonize

setup(
    name='IndexSet fast merge',
    ext_modules=cythonize("fast_merge.pyx")
)
