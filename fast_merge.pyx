# cython: language_level=3
# type: ignore

from libc.stdint cimport uint32_t

def merge(
        arr1: memoryview, start1: int, length1: int, offset1: int,
        arr2: memoryview, start2: int, length2: int, offset2: int,
        result: memoryview, take_first: bool, take_second: bool, take_common: bool,
    ) -> int:
    """
    Merge two sorted arrays A (arr1) and B (arr2), of unsigned (4-byte = 32-bit) integers.
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

    assert arr1.itemsize == arr2.itemsize == 4, "I can only handle 4-byte integer arrays."

    cdef uint32_t[::1] in1buffer = arr1
    cdef uint32_t[::1] in2buffer = arr2
    cdef uint32_t[::1] outbuffer = result

    cdef uint32_t* in1 = <uint32_t*> &in1buffer[0]
    cdef uint32_t* in2 = <uint32_t*> &in2buffer[0]
    cdef uint32_t* out = <uint32_t*> &outbuffer[0]

    return fast_merge(
        out,
        in1, start1, length1, offset1,
        in2, start2, length2, offset2,
        take_first, take_second, take_common,
    )


cdef size_t fast_merge(
        uint32_t* out,
        uint32_t* in1, size_t start1, size_t length1, size_t offset1,
        uint32_t* in2, size_t start2, size_t length2, size_t offset2,
        size_t take_first, size_t take_second, size_t take_common,
    ):

    in1 += start1
    in2 += start2

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0

    x = in1[i] - offset1
    y = in2[j] - offset2
    while True:
        if x < y:
            if take_first:
                out[k] = x
                k += 1
            i += 1
            if i >= length1:
                break
            x = in1[i] - offset1

        elif x > y:
            if take_second:
                out[k] = y
                k += 1
            j += 1
            if j >= length2:
                break
            y = in2[j] - offset2

        else:
            if take_common:
                out[k] = x
                k += 1
            i += 1
            if i >= length1:
                break
            j += 1
            if j >= length2:
                break
            x = in1[i] - offset1
            y = in2[j] - offset2

    if take_first:
        while i < length1:
            x = in1[i] - offset1
            i += 1
            out[k] = x
            k += 1

    if take_second:
        while j < length2:
            y = in2[j] - offset2
            j += 1
            out[k] = y
            k += 1

    return k
