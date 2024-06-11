# cython: language_level=3

def qsort_index(mview: memoryview, itemsize: int) -> None:
    """
    Sort an array of fixed-size bytestrings using C's builtin 'qsort'.
    The array is stored as an array of bytes/chars, and
    the size of each element is 'itemsize'.
    """
    cdef unsigned char[::1] buffer = mview
    cdef unsigned char* array = <unsigned char*> &buffer[0]
    size = len(mview) // itemsize
    if itemsize == 8:
        qsort(array, size, itemsize, compare8)
    elif itemsize == 12:
        qsort(array, size, itemsize, compare12)
    else:
        raise ValueError(f"Can only sort using itemsize 8 or 12, got size {itemsize}.")


cdef int compare8(const void* a, const void* b) noexcept:
    return memcmp(a, b, 8)

cdef int compare12(const void* a, const void* b) noexcept:
    return memcmp(a, b, 12)


cdef extern from "string.h":
    int memcmp(void *dest, void *src, size_t len)

cdef extern from "stdlib.h":
    void qsort(
        void *base, size_t count, size_t itemsize,
        int(*compare)(const void*, const void*)
    )

