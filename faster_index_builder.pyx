# cython: language_level=3
# type: ignore

from libc.stdint cimport uint8_t

from pathlib import Path
from mmap import mmap


def sort_index(path: Path, count: int, rowsize: int):
    with open(path, 'r+b') as file:
        mview = memoryview(mmap(file.fileno(), 0))
    sort_rows(mview, count, rowsize)

cdef void sort_rows(uint8_t[::1] view, size_t count, size_t rowsize):
    global sorting_rowsize
    sorting_rowsize = rowsize
    cdef uint8_t* array = <uint8_t*> &view[0]
    qsort(array, count, sorting_rowsize, compare)


###############################################################################
# Lowlevel functions for sorting and conversion

cdef size_t sorting_rowsize

cdef int compare(const void* a, const void* b) noexcept:
    global sorting_rowsize
    return memcmp(a, b, sorting_rowsize)

cdef extern from "string.h":
    int memcmp(void* dest, void* src, size_t len)

cdef extern from "stdlib.h":
    void qsort(
        void* base, size_t count, size_t rowsize,
        int(*compare)(const void*, const void*)
    )
