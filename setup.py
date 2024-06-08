from setuptools import setup
from Cython.Build import cythonize  # type: ignore

setup(
    name = 'IndexSet fast merge',
    ext_modules = cythonize([
        "fast_merge.pyx", 
        "multikey_quicksort.pyx",
        ]),  # type: ignore
)
