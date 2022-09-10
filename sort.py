
import sys
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
from typing import List, Callable, Any, MutableSequence, Union

from disk import DiskIntArrayType, DiskIntArray, DiskIntArrayBuilder
from util import ComparableWithCounter, progress_bar, setup_logger

SortableSequence = Union[DiskIntArrayType, MutableSequence[Any]]
SortKey = Callable[[Any], Any]
PivotSelector = Callable[[SortableSequence, int, int, SortKey], int]


###############################################################################
## Quicksort

def quicksort(array:SortableSequence, key:SortKey, pivotselector:PivotSelector, cutoff:int=0, debug:bool=False):
    """In-place quicksort."""
    if debug: 
        global _recursive_calls 
        _recursive_calls = 0
        ComparableWithCounter.ctr = 0
        debug_key = lambda n: ComparableWithCounter(key(n))
        logging.info(f"Quicksorting...")
        quicksort(array, debug_key, pivotselector, cutoff)
        logging.info(f"Summary: {_recursive_calls} recursive calls; {ComparableWithCounter.ctr} comparisons")
        return

    with progress_bar(total=len(array), desc="Quicksorting") as logger:
        quicksort_subarray(array, 0, len(array), key, pivotselector, cutoff, logger)
        logger.update(len(array) - logger.n)


_recursive_calls = 0

def quicksort_subarray(array:SortableSequence, lo:int, hi:int, key:SortKey, 
                       pivotselector:PivotSelector, cutoff:int, logger):
    """Quicksorts the subarray array[lo:hi] in place."""
    try:
        global _recursive_calls
        _recursive_calls += 1
    except NameError:
        pass
    logger.update(lo - logger.n)
    if hi <= lo + 1: 
        return
    if hi <= lo + cutoff:
        return builtin_timsort(array, lo, hi, key)
    mid = partition(array, lo, hi, key, pivotselector)
    quicksort_subarray(array, lo, mid, key, pivotselector, cutoff, logger)
    quicksort_subarray(array, mid+1, hi, key, pivotselector, cutoff, logger)


def builtin_timsort(array:SortableSequence, lo:int, hi:int, key:SortKey):
    """Call Python's built-in sort on the subarray array[lo:hi]."""
    sorted_array : List[int] = sorted(array[lo:hi], key=key)
    for i, val in enumerate(sorted_array, lo):
        array[i] = val


def partition(array:SortableSequence, lo:int, hi:int, 
              key:SortKey, pivotselector:PivotSelector) -> int:
    """Partition the subarray sa[lo:hi]. Returns the final index of the pivot."""
    i = pivotselector(array, lo, hi, key)
    if i != lo:
        array[lo], array[i] = array[i], array[lo]
    pivot = key(array[lo])

    i = lo + 1
    j = hi - 1
    while i <= j:
        while i <= j and key(array[i]) < pivot:
            i += 1
        while i <= j and pivot < key(array[j]):
            j -= 1
        if i <= j:
            array[i], array[j] = array[j], array[i]
            i += 1
            j -= 1

    array[lo], array[j] = array[j], array[lo]
    return j


def take_first_pivot(array:SortableSequence, lo:int, hi:int, key:SortKey) -> int:
    return lo


def random_pivot(array:SortableSequence, lo:int, hi:int, key:SortKey) -> int:
    return random.randrange(lo, hi)


def median_of_three(array:SortableSequence, lo:int, hi:int, key:SortKey) -> int:
    hi -= 1
    mid = (lo + hi) // 2
    return _median3(array, lo, mid, hi, key)


def tukey_ninther(array:SortableSequence, lo:int, hi:int, key:SortKey) -> int:
    N = hi - lo
    hi -= 1
    mid = lo + N//2
    delta = N//8
    m1 = _median3(array, lo, lo + delta, lo + 2*delta, key)
    m2 = _median3(array, mid - delta, mid, mid + delta, key)
    m3 = _median3(array, hi - 2*delta, hi - delta, hi, key)
    return _median3(array, m1, m2, m3, key)


def _median3(array:SortableSequence, i:int, j:int, k:int, key:SortKey) -> int:
    ti = key(array[i])
    tj = key(array[j])
    tk = key(array[k])
    if ti < tj:                 # ti < tj:
        if   tj < tk: return j  #   ti < tj < tk
        elif ti < tk: return k  #   ti < tk <= tj
        else:         return i  #   tk < ti < tj
    else:                       # tj <= ti:
        if   ti < tk: return i  #   tj <= ti < tk
        elif tj < tk: return k  #   tj < tk <= ti
        else:         return j  #   tk <= tj <= ti


###############################################################################
## Command line: testing the implementation

def check_sorted_array(array:SortableSequence, key:SortKey):
    for i, (left, right) in enumerate(zip(array, progress_bar(array[1:], desc="Checking array"))):
        assert key(left) < key(right), f"Ordering error in position {i}: {key(left)} >= {key(right)}"


def create_partially_shuffled_array(size:int, randomness:float, path:Path):
    """Fisher-Yates shuffle algorithm, but only swap elements 0-N times."""
    assert 0.0 <= randomness <= 1.0
    shuffles = round((size - 1) * randomness)
    with DiskIntArrayBuilder(path) as numbers:
        for i in progress_bar(range(size), desc="Creating array"):
            numbers.append(i)
    with DiskIntArray(path) as numbers:
        for i in progress_bar(range(shuffles), desc="Shuffling array"):
            index = round(i * size / shuffles)
            other = random.randrange(index, size)
            numbers[index], numbers[other] = numbers[other], numbers[index]


def main(args:Namespace):
    if args.recursion_limit:
        sys.setrecursionlimit(args.recursion_limit)
    create_partially_shuffled_array(args.num, args.randomness, args.array_path)

    sortkey = lambda n:n  # the identity function
    with DiskIntArray(args.array_path) as numbers:
        pivot = (
            take_first_pivot if args.pivot == "take-first"      else 
            random_pivot     if args.pivot == "random"          else
            median_of_three  if args.pivot == "median-of-three" else 
            tukey_ninther    if args.pivot == "tukey-ninther"   else 
            NotImplemented
        )
        quicksort(numbers, sortkey, pivot, cutoff=args.cutoff, debug=args.debug)
        check_sorted_array(numbers, sortkey)


default_array_path = Path('testarray.tmp')

parser = ArgumentParser(description='Testing different sorting implementations')
parser.add_argument('--num', '-n', type=int, default=100_000,
    help=f'sort the numbers 0...N-1 (default: 100,000)')
parser.add_argument('--randomness', '-r', type=float, default=1.0,
    help=f'randomness in shuffling the numbers (default: 1.0)')
parser.add_argument('--pivot', '-p', default='take-first', 
    choices=['take-first', 'random', 'median-of-three', 'tukey-ninther'], 
    help=f'pivot selector (default: take-first)')
parser.add_argument('--cutoff', '-c', type=int, default=0,
    help=f'cutoff to built-in Timsort (default: 0)')
parser.add_argument('--recursion-limit', '-R', type=int,
    help=f"set the recursion limit in Python (default: {sys.getrecursionlimit()})")
parser.add_argument('--array-path', type=Path, default=default_array_path,
    help=f"the path to the external array file (default: {default_array_path})")
parser.add_argument('--debug', action='store_true',
    help=f"debugging")


if __name__ == '__main__':
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=logging.DEBUG)
    main(parser.parse_args())
