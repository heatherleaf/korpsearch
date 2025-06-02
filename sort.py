
import random
from mmap import mmap

from util import progress_bar, ProgressBar


BytesArray = mmap | bytearray
ArrayValue = bytes | bytearray


###############################################################################
## Quicksort

def quicksort(array: BytesArray, itemsize: int, cutoff: int = 1000) -> None:
    """In-place quicksort."""
    assert len(array) % itemsize == 0
    assert cutoff >= 10
    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    hi = len(array) // itemsize
    logger: ProgressBar[None]
    with progress_bar(total=hi, desc="Quicksorting") as logger:  # type: ignore
        quicksort_subarray(array, itemsize, 0, hi, cutoff, logger)
        logger.update(hi - logger.n)
    sys.setrecursionlimit(lim)


def quicksort_subarray(array: BytesArray, w: int, lo: int, hi: int, cutoff: int, logger: ProgressBar[None]) -> None:
    """Quicksorts the subarray array[lo:hi] in place."""
    logger.update(lo - logger.n)
    if hi - lo <= cutoff:
        builtin_timsort(array, w, lo, hi)
    else:
        mid = partition(array, w, lo, hi)
        quicksort_subarray(array, w, lo, mid, cutoff, logger)
        quicksort_subarray(array, w, mid+1, hi, cutoff, logger)


def builtin_timsort(array: BytesArray, w: int, lo: int, hi: int) -> None:
    """Call Python's built-in sort on the subarray array[lo:hi]."""
    sorted_array: list[ArrayValue] = []
    for i in range(lo, hi):
        sorted_array.append(get_value(array, w, i))
    sorted_array.sort()
    for i, val in enumerate(sorted_array, lo):
        set_value(array, w, i, val)


def partition(array: BytesArray, w: int, lo: int, hi: int) -> int:
    """Partition the subarray sa[lo:hi]. Returns the final index of the pivot."""
    p = random.randrange(lo, hi)
    swap_values(array, w, lo, p)
    pivot = get_value(array, w, lo)

    i = lo + 1
    j = hi - 1
    while i <= j:
        while i <= j and get_value(array, w, i) < pivot:
            i += 1
        while i <= j and pivot < get_value(array, w, j):
            j -= 1
        if i <= j:
            swap_values(array, w, i, j)
            i += 1
            j -= 1

    swap_values(array, w, lo, j)
    return j


###############################################################################
## Getting, setting and swapping values

def get_value(array: BytesArray, w: int, i: int) -> ArrayValue:
    return array[i*w : (i+1)*w]

def set_value(array: BytesArray, w: int, i: int, val: ArrayValue) -> None:
    array[i*w : (i+1)*w] = val

def swap_values(array: BytesArray, w: int, i: int, j: int) -> None:
    array[i*w : (i+1)*w], array[j*w : (j+1)*w] = array[j*w : (j+1)*w], array[i*w : (i+1)*w]

