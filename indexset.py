
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
    values : IndexSetValuesType

    def __init__(self, values:IndexSetValuesType, start:int=0, size:int=-1):
        self.values = values
        self.start = start
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
        yield from self.values[self.start:self.start+self.size]

    def intersection_update(self, other:'IndexSet'):
        assert not isinstance(self.values, list) \
            and not isinstance(other.values, list)

        self.values = self.intersection(other)
        self.start = 0
        self.size = len(self.values)

    def intersection(self, other:'IndexSet') -> List[int]:
        """Take the intersection of two sorted arrays."""
        arr1, start1, length1 = self.values, self.start, self.size
        arr2, start2, length2 = other.values, other.start, other.size
        try:
            return fast_intersection.intersection(arr1, start1, length1, arr2, start2, length2)
        except NameError:
            pass
        result = []
        k1, k2 = 0, 0
        x1, x2 = arr1[start1], arr2[start2]
        try:
            while True:
                if x1 < x2: 
                    k1 += 1
                    x1 = arr1[start1 + k1]
                elif x1 > x2:
                    k2 += 1
                    x2 = arr2[start2 + k2]
                else:
                    result.append(x1)
                    k1 += 1
                    x1 = arr1[start1 + k1]
                    k2 += 1
                    x2 = arr2[start2 + k2]
        except IndexError:
            pass
        return result

    def filter(self, check:Callable[[int],bool]):
        self.values = [elem for elem in self if check(elem)]
        self.start = 0
        self.size = len(self.values)

    def __contains__(self, elem:int) -> bool:
        values = self.values
        start : int = self.start
        end : int = start + self.size - 1
        while start <= end:
            mid : int = (start + end) // 2
            mid_elem : int = values[mid]
            if mid_elem == elem:
                return True
            elif mid_elem < elem:
                start = mid + 1
            else:
                end = mid - 1
        return False

