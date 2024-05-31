
import random
from collections.abc import Callable, MutableSequence

from util import progress_bar, ProgressBar, T

PivotSelector = Callable[[MutableSequence[T], int, int], int]


###############################################################################
## Quicksort

def quicksort(array: MutableSequence[T], pivotselector: PivotSelector[T], cutoff: int = 0) -> None:
    """In-place quicksort."""
    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    logger: ProgressBar[T]
    with progress_bar(total=len(array), desc="Quicksorting") as logger:
        quicksort_subarray(array, 0, len(array), pivotselector, cutoff, logger)
        logger.update(len(array) - logger.n)
    sys.setrecursionlimit(lim)


def quicksort_subarray(array: MutableSequence[T], lo: int, hi: int, 
                       pivotselector: PivotSelector[T], cutoff: int, logger: ProgressBar[T]) -> None:
    """Quicksorts the subarray array[lo:hi] in place."""
    logger.update(lo - logger.n)
    size = hi - lo
    if size == 2:
        if array[lo] > array[lo+1]:  # type: ignore
            array[lo], array[lo+1] = array[lo+1], array[lo]
    elif size <= cutoff:
        builtin_timsort(array, lo, hi)
    elif size > 2:
        mid = partition(array, lo, hi, pivotselector)
        quicksort_subarray(array, lo, mid, pivotselector, cutoff, logger)
        quicksort_subarray(array, mid+1, hi, pivotselector, cutoff, logger)


def builtin_timsort(array: MutableSequence[T], lo: int, hi: int) -> None:
    """Call Python's built-in sort on the subarray array[lo:hi]."""
    sorted_array: list[T] = sorted(array[lo:hi])  # type: ignore
    for i, val in enumerate(sorted_array, lo):
        array[i] = val


def partition(array: MutableSequence[T], lo: int, hi: int, pivotselector: PivotSelector[T]) -> int:
    """Partition the subarray sa[lo:hi]. Returns the final index of the pivot."""
    i = pivotselector(array, lo, hi)
    if i != lo:
        array[lo], array[i] = array[i], array[lo]
    pivot = array[lo]

    i = lo + 1
    j = hi - 1
    while i <= j:
        while i <= j and array[i] < pivot:  # type: ignore
            i += 1
        while i <= j and pivot < array[j]:  # type: ignore
            j -= 1
        if i <= j:
            array[i], array[j] = array[j], array[i]
            i += 1
            j -= 1

    array[lo], array[j] = array[j], array[lo]
    return j


def take_first_pivot(array: MutableSequence[T], lo: int, hi: int) -> int:
    return lo


def random_pivot(array: MutableSequence[T], lo: int, hi: int) -> int:
    return random.randrange(lo, hi)


def median_of_three(array: MutableSequence[T], lo: int, hi: int) -> int:
    hi -= 1
    mid = (lo + hi) // 2
    return _median3(array, lo, mid, hi)


def tukey_ninther(array: MutableSequence[T], lo: int, hi: int) -> int:
    N = hi - lo
    hi -= 1
    mid = lo + N//2
    delta = N//8
    m1 = _median3(array, lo, lo + delta, lo + 2*delta)
    m2 = _median3(array, mid - delta, mid, mid + delta)
    m3 = _median3(array, hi - 2*delta, hi - delta, hi)
    return _median3(array, m1, m2, m3)


def _median3(array: MutableSequence[T], i: int, j: int, k: int) -> int:
    ti = array[i]
    tj = array[j]
    tk = array[k]
    if ti < tj:                 # type: ignore  # ti < tj:
        if   tj < tk: return j  # type: ignore  #   ti < tj < tk
        elif ti < tk: return k  # type: ignore  #   ti < tk <= tj
        else:         return i  #               #   tk < ti < tj
    else:                       #               # tj <= ti:
        if   ti < tk: return i  # type: ignore  #   tj <= ti < tk
        elif tj < tk: return k  # type: ignore  #   tj < tk <= ti
        else:         return j  #               #   tk <= tj <= ti
