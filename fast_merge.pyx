# cython: language_level=3

def merge(
        arr1: memoryview, start1: int, length1: int, offset1: int, 
        arr2: memoryview, start2: int, length2: int, offset2: int, 
        result: memoryview, take_first: bool, take_second: bool, take_common: bool,
    ) -> int:
    """
    Merge two sorted arrays A (arr1) and B (arr2).
    The result array must have enough space for all elements.
    Returns the size of the merged result (so the result array can be truncated).

    * If take_first  is True then elements in A - B are included.
    * If take_second is True then elements in B - A are included.
    * If take_common is True then elements in A & B are included.

    You can get the following set operations (among others):

    Operation              take_first     take_second    take_common
    union        (A | B)   True           True           True
    intersection (A & B)   False          False          True
    difference   (A - B)   True           False          False
    """

    assert arr1.itemsize == arr2.itemsize
    cdef int itemsize = arr1.itemsize

    cdef unsigned int[::1] in1buffer = arr1
    cdef unsigned int[::1] in2buffer = arr2
    cdef unsigned int[::1] outbuffer = result

    cdef unsigned char* in1 = <unsigned char*> &in1buffer[0]
    cdef unsigned char* in2 = <unsigned char*> &in2buffer[0]
    cdef unsigned char* out = <unsigned char*> &outbuffer[0]

    return fast_merge(
        out, itemsize, 
        in1, start1, length1, offset1,
        in2, start2, length2, offset2,
        take_first, take_second, take_common,
    )


cdef size_t fast_merge(
        unsigned char* out, size_t itemsize, 
        unsigned char* in1, size_t start1, size_t length1, size_t offset1, 
        unsigned char* in2, size_t start2, size_t length2, size_t offset2, 
        size_t take_first, size_t take_second, size_t take_common,
    ):

    in1 += start1 * itemsize
    in2 += start2 * itemsize
    length1 *= itemsize
    length2 *= itemsize

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0

    x = read_bytes(in1 + i, itemsize) - offset1
    y = read_bytes(in2 + j, itemsize) - offset2
    while True:
        if x < y: 
            if take_first:
                write_bytes(out + k, x, itemsize)
                k += itemsize
            i += itemsize
            if i >= length1: 
                break
            x = read_bytes(in1 + i, itemsize) - offset1

        elif x > y: 
            if take_second:
                write_bytes(out + k, y, itemsize)
                k += itemsize
            j += itemsize
            if j >= length2: 
                break
            y = read_bytes(in2 + j, itemsize) - offset2

        else:
            if take_common:
                write_bytes(out + k, x, itemsize)
                k += itemsize
            i += itemsize
            if i >= length1: 
                break
            j += itemsize
            if j >= length2: 
                break
            x = read_bytes(in1 + i, itemsize) - offset1
            y = read_bytes(in2 + j, itemsize) - offset2

    if take_first:
        while i < length1:
            x = read_bytes(in1 + i, itemsize) - offset1
            i += itemsize
            write_bytes(out + k, x, itemsize)
            k += itemsize

    if take_second:
        while j < length2:
            y = read_bytes(in2 + j, itemsize) - offset2
            j += itemsize
            write_bytes(out + k, y, itemsize)
            k += itemsize

    return k // itemsize


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
