# cython: language_level=3
# type: ignore

from libc.stdint cimport uint8_t, uint32_t
from libc.stdio cimport FILE

from pathlib import Path
from mmap import mmap
from argparse import Namespace
import logging

from disk import DiskIntArray


# Note: FasterCollector.append2/3 does not work well together with PyPy,
# probably it's not good to switch between Python and Cython often
# because it break the JIT compilation.


cdef class FasterCollector:
    args: Namespace
    tmppath: Path
    cdef FILE* tmpfile
    cdef size_t rowsize

    def __init__(self, rowsize: int, tmppath: Path, args: Namespace):
        self.rowsize = rowsize
        self.args = args
        self.tmppath = tmppath
        self.tmpfile = fopen(bytes(tmppath), 'w')

    cpdef void append2(self, uint32_t a, uint32_t b):
        # assert self.rowsize == 2
        fwrite_value(a, self.tmpfile)
        fwrite_value(b, self.tmpfile)

    cpdef void append3(self, uint32_t a, uint32_t b, uint32_t c):
        # assert self.rowsize == 2
        fwrite_value(a, self.tmpfile)
        fwrite_value(b, self.tmpfile)
        fwrite_value(c, self.tmpfile)

    cpdef void append(self, values: tuple[uint32_t, ...]):
        # assert len(values) == self.rowsize
        cdef uint32_t val
        for val in values:
            fwrite_value(val, self.tmpfile)

    def finalise(self, index_path: Path):
        cdef size_t nr_rows = ftell(self.tmpfile) // (self.rowsize * 4)
        fclose(self.tmpfile)
        with open(self.tmppath, 'r+b') as file:
            tmpview = memoryview(mmap(file.fileno(), 0))

        logging.debug(f"Sorting {nr_rows} rows using C 'qsort'")
        sort_index(tmpview, nr_rows, self.rowsize)

        logging.debug(f"Creating index file: {index_path}")
        with DiskIntArray.create(nr_rows, index_path) as suffix_array:
            # Turn the byte-view into a view of 4-byte unsigned ints (uint32).
            create_index_array(tmpview.cast('I'), suffix_array, nr_rows, self.rowsize)

        if not self.args.keep_tmpfiles:
            self.tmppath.unlink()


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
# Lowlevel functions for sorting

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


###############################################################################
# Lowlevel functions for binary file handling

cdef inline void fwrite_value(uint32_t val, FILE* file):
    cdef uint32_t nval = htonl(val)
    fwrite(&nval, sizeof(uint32_t), 1, file)

cdef extern from "stdio.h":
    size_t fwrite(const void* buffer, size_t size, size_t count, FILE* stream)
    FILE* fopen(const char* filename, const char* mode)
    int fclose(FILE* stream)
    long ftell(FILE* stream)

cdef extern from "arpa/inet.h":
    uint32_t ntohl(uint32_t netlong)
    uint32_t htonl(uint32_t netlong)

