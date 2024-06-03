# cython: language_level=3

import sys
from libc.stdlib cimport malloc, free

from disk import LowlevelIntArray


ctypedef const unsigned char[::1] buffer


# TODO:
#  1. Cython has memoryviews too, so it should be possible to use them directly 
#     instead of working with byte buffers.
#  2. We want to be able to update a set in-place of creating a new every time,
#     just as is already done in Python (see IndexSet._merge_internal)


cdef buffer to_buffer(array, size_t start, size_t length, size_t size):
    """Convert an array to an array of bytes."""
    start *= size
    length *= size
    return array._bytearray[start : start+length]


def merge(arr1, start1, length1, offset1, arr2, start2, length2, offset2, take_first, take_second, take_common):
    """Merge two sorted arrays A and B.

    * If take_first is True then elements in A-B are included.
    * If take_second is True then elements in B-A are included.
    * If take_common is True then elements in intersection(A,B) are included.

    You can get the following set operations (among others):

    Operation           take_first     take_second    take_common
    union(A,B)          True           True           True
    intersection(A,B)   False          False          True
    A-B                 True           False          False
    """

    assert arr1.itemsize == arr2.itemsize
    cdef int itemsize = arr1.itemsize

    cdef buffer buf1 = to_buffer(arr1, start1, length1, itemsize)
    cdef buffer buf2 = to_buffer(arr2, start2, length2, itemsize)
    # TODO: 
    # if not (take_first or take_second): we can use outsize = min(len1, len2))
    # if not take_second: we can use outsize = len1
    # if not take_first: we can use outsize = len2
    cdef size_t outsize = buf1.nbytes + buf2.nbytes
    cdef unsigned char* out = <unsigned char*> malloc(outsize)

    k = fast_merge(out, itemsize, 
                   &buf1[0], start1, buf1.nbytes, offset1,
                   &buf2[0], start2, buf2.nbytes, offset2,
                   take_first, take_second, take_common)

    result = LowlevelIntArray(out[:k], itemsize)
    free(out)
    return result


cdef size_t fast_merge(unsigned char* out, size_t itemsize, 
                       const unsigned char* in1, size_t start1, size_t len1, size_t offset1, 
                       const unsigned char* in2, size_t start2, size_t len2, size_t offset2, 
                       size_t take_first, size_t take_second, size_t take_common):

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0

    while i < len1 and j < len2:
        x = read_bytes(in1+i, itemsize) - offset1
        y = read_bytes(in2+j, itemsize) - offset2

        if x < y: 
            i += itemsize
            if take_first:
                write_bytes(out+k, x, itemsize)
                k += itemsize
        elif x > y: 
            j += itemsize
            if take_second:
                write_bytes(out+k, y, itemsize)
                k += itemsize
        else:
            i += itemsize
            j += itemsize
            if take_common:
                write_bytes(out+k, x, itemsize)
                k += itemsize

    if take_first:
        memcpy(out+k, in1+i, len1-i)
        k += len1-i
    if take_second:
        memcpy(out+k, in2+j, len2-j)
        k += len2-j

    return k


cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t len)


cdef inline size_t read_bytes(const void *ptr, size_t size):
    """Read an integer of the given number of bytes from a pointer."""
    cdef size_t result = 0
    memcpy(&result, ptr, size)
    return result


cdef inline void write_bytes(void *ptr, size_t value, size_t size):
    """Write an integer of the given number of bytes to a pointer."""
    memcpy(ptr, &value, size)
