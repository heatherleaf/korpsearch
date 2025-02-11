# cython: language_level=3
# type: ignore

from libc.stdint cimport uint8_t, uint32_t

from pathlib import Path
from mmap import mmap
import logging

from disk import DiskIntArray


def finalise(tmppath: Path, nr_rows: int, rowsize: int, index_path: Path):
    with open(tmppath, 'r+b') as file:
        tmpview = memoryview(mmap(file.fileno(), 0))

    logging.debug(f"Sorting {nr_rows} rows using C 'qsort'")
    sort_index(tmpview, nr_rows, rowsize)

    logging.debug(f"Creating index file: {index_path}")
    with DiskIntArray.create(nr_rows, index_path) as suffix_array:
        # Turn the byte-view into a view of 4-byte unsigned ints (uint32)
        create_index_array(tmpview.cast('I'), suffix_array, nr_rows, rowsize)


cdef void sort_index(
            uint8_t[::1] sorted_view,
            size_t nr_rows,
            size_t rowsize,
        ):
    global sorting_itemsize
    sorting_itemsize = 4 * rowsize
    cdef uint8_t* sorted_array = <uint8_t*> &sorted_view[0]
    qsort(sorted_array, nr_rows, sorting_itemsize, compare)


cdef void create_index_array(
            uint32_t[::1] sorted_view,
            uint32_t[::1] index_view,
            size_t nr_rows,
            size_t rowsize,
        ):
    cdef uint32_t* sorted_array = <uint32_t*> &sorted_view[0]
    cdef uint32_t* index_array = <uint32_t*> &index_view[0]
    cdef size_t i
    for i in range(nr_rows):
        index_array[i] = ntohl(sorted_array[(i + 1) * rowsize - 1])


###############################################################################
# Lowlevel functions for sorting and conversion

cdef size_t sorting_itemsize

cdef int compare(const void* a, const void* b) noexcept:
    global sorting_itemsize
    return memcmp(a, b, sorting_itemsize)

cdef extern from "string.h":
    int memcmp(void* dest, void* src, size_t len)

cdef extern from "stdlib.h":
    void qsort(
        void* base, size_t count, size_t rowsize,
        int(*compare)(const void*, const void*)
    )

cdef extern from "arpa/inet.h":
    uint32_t ntohl(uint32_t netlong)

