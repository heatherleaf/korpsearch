# cython: language_level=3

from libc.stdlib cimport malloc


def multikey_quicksort(mview: memoryview, itemsize: int) -> None:
    """
    Sort an array of fixed-size bytestrings using multikey quicksort:
    https://en.wikipedia.org/wiki/Multi-key_quicksort
    The array is stored as an array of bytes/chars, and
    the size of each element is 'itemsize'.
    """
    cdef unsigned char[::1] buffer = mview
    cdef unsigned char* array = <unsigned char*> &buffer[0]
    # Temporary bytestring used for swapping:
    cdef unsigned char* tmp = <unsigned char*> malloc(itemsize)
    fast_mk_quicksort(array, tmp, itemsize, 0, len(mview) // itemsize, 0)


cdef void fast_mk_quicksort(
        unsigned char* array, unsigned char* tmp, size_t itemsize, 
        size_t start, size_t end, size_t offset,
    ):
    if end - start <= 1:
        return
    if end - start > 100_000:
        print("Sorted:", start, "rows", end="\r")

    # Use the very simple take-first pivot,
    # so no need to swap the pivot into the first position.
    cdef unsigned char pivotChar = getCharAtOffset(array, itemsize, start, offset)

    # Partition the array in three parts, <, == and >
    cdef size_t pivotStart = start
    cdef size_t pivotEnd = end
    cdef size_t i = pivotStart + 1
    while i < pivotEnd:
        thisChar = getCharAtOffset(array, itemsize, i, offset)
        if thisChar == pivotChar:
            i += 1
        elif thisChar < pivotChar:
            swap(array, tmp, itemsize, pivotStart, i)
            pivotStart += 1
            i += 1
        else: # thisChar > pivotChar:
            pivotEnd -= 1
            swap(array, tmp, itemsize, i, pivotEnd)

    fast_mk_quicksort(array, tmp, itemsize, start, pivotStart, offset)
    fast_mk_quicksort(array, tmp, itemsize, pivotStart, pivotEnd, offset+1)
    fast_mk_quicksort(array, tmp, itemsize, pivotEnd, end, offset)


cdef unsigned char getCharAtOffset(unsigned char* array, size_t itemsize, size_t i, size_t offset):
    return array[i * itemsize + offset]


cdef inline void swap(unsigned char* array, unsigned char* tmp, size_t itemsize, size_t i, size_t j):
    memcpy(tmp, array + i*itemsize, itemsize)
    memcpy(array + i*itemsize, array + j*itemsize, itemsize)
    memcpy(array + j*itemsize, tmp, itemsize)


cdef extern from "string.h":
    void *memcpy(void *dest, void *src, size_t len)

