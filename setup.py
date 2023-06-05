from setuptools import setup
from Cython.Build import cythonize

setup(
    name='IndexSet fast intersection',
    ext_modules=cythonize("fast_intersection.pyx")
)
