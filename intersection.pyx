# cython: language_level=3

from disk import SlowDiskIntArray
import sys

from libc.stdlib cimport malloc, free

ctypedef const unsigned char[::1] buffer

cdef buffer to_buffer(array, start, length):
    """Convert an array to an array of bytes."""

    cdef buffer result
    start *= elemsize(array)
    length *= elemsize(array)

    if isinstance(array, memoryview):
        result = array.obj
    elif isinstance(array, SlowDiskIntArray):
        assert array._byteorder == sys.byteorder
        result = array._array
    else:
        assert False, "argument to to_memoryview has unknown type"

    return result[start:start+length]

cdef elemsize(array):
    """Find the element size of an array."""

    if isinstance(array, memoryview):
        return array.itemsize
    elif isinstance(array, SlowDiskIntArray):
        return array._elemsize
    else:
        assert False, "argument to elemsize has unknown type"

def intersection(arr1, start1, length1, arr2, start2, length2):
    """Take the intersection of two sorted arrays."""

    assert elemsize(arr1) == elemsize(arr2)
    size = elemsize(arr1)

    cdef buffer buf1 = to_buffer(arr1, start1, length1)
    cdef buffer buf2 = to_buffer(arr2, start2, length2)
    out = <char*>malloc(max(len(buf1), len(buf2)))

    try:
        length = intersection_switch(&buf1[0], len(buf1), &buf2[0], len(buf2), out, size)
        return SlowDiskIntArray(out[:length], size, sys.byteorder)
    finally:
        free(out)

cdef int intersection_switch(const void *in1, int len1, const void *in2, int len2, void *out, int size):
    # Generate specialised code for each value of 'size'.
    # This improves performance because it allows the C compiler to specialise
    # read_bytes and write_bytes to the given size.
    if size == 1: return intersection_core(in1, len1, in2, len2, out, 1)
    elif size == 2: return intersection_core(in1, len1, in2, len2, out, 2)
    elif size == 3: return intersection_core(in1, len1, in2, len2, out, 3)
    elif size == 4: return intersection_core(in1, len1, in2, len2, out, 4)
    else: return intersection_core(in1, len1, in2, len2, out, size)

cdef inline int intersection_core(const void *in1, int len1, const void *in2, int len2, void *out, int size):
    """The low-level intersection routine."""

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    while i < len1 and j < len2:
        x = read_bytes(in1+i, size)
        y = read_bytes(in2+j, size)

        if x < y: i += size
        elif x > y: j += size
        else:
            write_bytes(out+k, x, size)
            i += size
            j += size
            k += size
    return k

cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t len)

cdef inline long read_bytes(const void *ptr, int size):
    """Read an integer of the given number of bytes from a pointer."""

    cdef long result = 0
    memcpy(&result, ptr, size)
    return result

cdef inline void write_bytes(void *ptr, long value, int size):
    """Write an integer of the given number of bytes to a pointer."""

    memcpy(ptr, &value, size)
