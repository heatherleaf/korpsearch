
import sys
import itertools
from typing import List, Iterator, Union, Callable

from disk import DiskIntArrayType

try:
    import fast_intersection  # type: ignore
except ModuleNotFoundError:
    print("Module 'fast_intersection' not found.\n"
          "To install, run: 'python setup.py build_ext --inplace'.\n"
          "Using a slow internal implementation instead.\n", 
          file=sys.stderr)


################################################################################
## Index set

IndexSetValuesType = Union[DiskIntArrayType, List[int]]

class IndexSet:
    start : int
    size : int
    offset : int
    values : IndexSetValuesType

    def __init__(self, values:IndexSetValuesType, start:int=0, size:int=-1, offset:int=0):
        self.values = values
        self.start = start
        self.offset = offset
        self.size = size
        if isinstance(values, list) and size < 0:
            self.size = len(values) - start
        else:
            assert size > 0

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        MAX = 5
        if len(self) <= MAX:
            return "{" + ", ".join(str(n) for n in self) + "}"
        return f"{{{', '.join(str(n) for n in itertools.islice(self, MAX))}, ... (N={len(self)})}}"

    def __iter__(self) -> Iterator[int]:
        offset = self.offset
        for val in self.values[self.start:self.start+self.size]:
            yield val - offset

    def intersection_update(self, other:'IndexSet'):
        self.values = self.intersection(other)
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    def intersection(self, other:'IndexSet') -> List[int]:
        """Take the intersection of two sorted arrays."""
        arr1, start1, length1, offset1 = self.values, self.start, self.size, self.offset
        arr2, start2, length2, offset2 = other.values, other.start, other.size, other.offset
        if not (isinstance(self.values, list) or offset1 > 0 or 
                isinstance(other.values, list) or offset2 > 0):
            try:
                return fast_intersection.intersection(
                    arr1, start1, length1, # offset1,
                    arr2, start2, length2, # offset2,
                )
            except NameError:
                pass

        result = []
        k1, k2 = 0, 0
        x1, x2 = arr1[start1] - offset1, arr2[start2] - offset2
        try:
            while True:
                if x1 < x2: 
                    k1 += 1
                    x1 = arr1[start1 + k1] - offset1
                elif x1 > x2:
                    k2 += 1
                    x2 = arr2[start2 + k2] - offset2
                else:
                    result.append(x1)
                    k1 += 1
                    x1 = arr1[start1 + k1] - offset1
                    k2 += 1
                    x2 = arr2[start2 + k2] - offset2
        except IndexError:
            pass
        return result

    def filter(self, check:Callable[[int],bool]):
        self.values = [elem for elem in self if check(elem)]
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    def __contains__(self, elem:int) -> bool:
        values = self.values
        offset = self.offset
        start = self.start
        end = start + self.size - 1
        while start <= end:
            mid = (start + end) // 2
            mid_elem = values[mid] - offset
            if mid_elem == elem:
                return True
            elif mid_elem < elem:
                start = mid + 1
            else:
                end = mid - 1
        return False

