# cython: language_level=3

from disk import LowlevelIntArray
import sys

from libc.stdlib cimport malloc, free

ctypedef const unsigned char[::1] buffer


cdef buffer to_buffer(array, start, length):
    """Convert an array to an array of bytes."""
    assert array._byteorder == sys.byteorder
    start *= array._elemsize
    length *= array._elemsize
    return array._mmap[start : start+length]

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

    assert arr1._elemsize == arr2._elemsize
    cdef int size = arr1._elemsize

    cdef buffer buf1 = to_buffer(arr1, start1, length1)
    cdef buffer buf2 = to_buffer(arr2, start2, length2)

    cdef const unsigned char* in1 = &buf1[0]
    cdef const unsigned char* in2 = &buf2[0]
    cdef size_t len1 = buf1.nbytes
    cdef size_t len2 = buf2.nbytes

    out = <char*>malloc(len1 + len2)

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0

    while i < len1 and j < len2:
        x = read_bytes(in1+i, size) - offset1
        y = read_bytes(in2+j, size) - offset2

        if x < y: 
            i += size
            if take_first:
                write_bytes(out+k, x, size)
                k += size
        elif x > y: 
            j += size
            if take_second:
                write_bytes(out+k, y, size)
                k += size
        else:
            i += size
            j += size
            if take_common:
                write_bytes(out+k, x, size)
                k += size

    if take_first:
        memcpy(out+k, in1+i, len1-i)
        k += len1-i
    if take_second:
        memcpy(out+k, in2+j, len2-j)
        k += len2-j

    result = LowlevelIntArray(bytemap=out[:k], elemsize=size, byteorder=sys.byteorder)
    free(out)
    return result


cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t len)


cdef inline size_t read_bytes(const void *ptr, int size):
    """Read an integer of the given number of bytes from a pointer."""
    cdef size_t result = 0
    memcpy(&result, ptr, size)
    return result


cdef inline void write_bytes(void *ptr, size_t value, int size):
    """Write an integer of the given number of bytes to a pointer."""
    memcpy(ptr, &value, size)
