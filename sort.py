
import random
from mmap import mmap
from typing import Union
from collections.abc import Callable

from util import progress_bar, ProgressBar


BytesArray = Union[mmap, bytearray]
ArrayValue = Union[bytes, bytearray]

PivotSelector = Callable[[BytesArray, int, int, int], int]


###############################################################################
## Quicksort

def quicksort(array: BytesArray, itemsize: int, pivotselector: PivotSelector, cutoff: int = 1000) -> None:
    """In-place quicksort."""
    assert len(array) % itemsize == 0
    assert cutoff >= 10
    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    hi = len(array) // itemsize
    logger: ProgressBar[None]
    with progress_bar(total=hi, desc="Quicksorting") as logger:
        quicksort_subarray(array, itemsize, 0, hi, pivotselector, cutoff, logger)
        logger.update(hi - logger.n)
    sys.setrecursionlimit(lim)


def quicksort_subarray(array: BytesArray, w: int, lo: int, hi: int, 
                       pivotselector: PivotSelector, cutoff: int, logger: ProgressBar[None]) -> None:
    """Quicksorts the subarray array[lo:hi] in place."""
    logger.update(lo - logger.n)
    if hi - lo <= cutoff:
        builtin_timsort(array, w, lo, hi)
    else:
        mid = partition(array, w, lo, hi, pivotselector)
        quicksort_subarray(array, w, lo, mid, pivotselector, cutoff, logger)
        quicksort_subarray(array, w, mid+1, hi, pivotselector, cutoff, logger)


def builtin_timsort(array: BytesArray, w: int, lo: int, hi: int) -> None:
    """Call Python's built-in sort on the subarray array[lo:hi]."""
    sorted_array: list[ArrayValue] = []
    for i in range(lo, hi):
        sorted_array.append(get_value(array, w, i))
    sorted_array.sort()
    for i, val in enumerate(sorted_array, lo):
        set_value(array, w, i, val)


def partition(array: BytesArray, w: int, lo: int, hi: int, pivotselector: PivotSelector) -> int:
    """Partition the subarray sa[lo:hi]. Returns the final index of the pivot."""
    i = pivotselector(array, w, lo, hi)
    if i != lo:
        swap_values(array, w, lo, i)
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


###############################################################################
## Pivot selectors

def take_first_pivot(array: BytesArray, w: int, lo: int, hi: int) -> int:
    return lo


def take_middle_pivot(array: BytesArray, w: int, lo: int, hi: int) -> int:
    return (lo + hi) // 2


def random_pivot(array: BytesArray, w: int, lo: int, hi: int) -> int:
    return random.randrange(lo, hi)


def median_of_three(array: BytesArray, w: int, lo: int, hi: int) -> int:
    hi -= w
    mid = (lo + hi) // 2
    return _median3(array, w, lo, mid, hi)


def tukey_ninther(array: BytesArray, w: int, lo: int, hi: int) -> int:
    N = hi - lo
    hi -= w
    mid = lo + N//2
    delta = N//8
    m1 = _median3(array, w, lo, lo + delta, lo + 2*delta)
    m2 = _median3(array, w, mid - delta, mid, mid + delta)
    m3 = _median3(array, w, hi - 2*delta, hi - delta, hi)
    return _median3(array, w, m1, m2, m3)


def _median3(array: BytesArray, w: int, i: int, j: int, k: int) -> int:
    ti = get_value(array, w, i)
    tj = get_value(array, w, j)
    tk = get_value(array, w, k)
    if ti < tj:                 # ti < tj:
        if   tj < tk: return j  #   ti < tj < tk
        elif ti < tk: return k  #   ti < tk <= tj
        else:         return i  #   tk < ti < tj
    else:                       # tj <= ti:
        if   ti < tk: return i  #   tj <= ti < tk
        elif tj < tk: return k  #   tj < tk <= ti
        else:         return j  #   tk <= tj <= ti
