
import sys
import itertools
from typing import List, Iterator, Union, Callable

from disk import DiskIntArray

try:
    import platform
    if platform.python_implementation() == 'CPython':
        # fast_intersection is NOT faster in PyPy!
        import fast_intersection  # type: ignore
except ModuleNotFoundError:
    print("Module 'fast_intersection' not found. "
          "To install, run: 'python setup.py build_ext --inplace'. "
          "Defaulting to an internal implementation.\n", 
          file=sys.stderr)


################################################################################
## Index set

IndexSetValuesType = Union[DiskIntArray, List[int]]

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
        while self.values[self.start] < self.offset:
            self.start += 1
            self.size -= 1
        assert self.size > 0

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

    def intersection_update(self, other:'IndexSet', use_internal:bool=False):
        self.values = self.intersection(other, use_internal=use_internal)
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    def intersection(self, other:'IndexSet', use_internal:bool=False) -> List[int]:
        """Take the intersection of two sorted arrays."""
        if (isinstance(self.values, DiskIntArray) and 
            isinstance(other.values, DiskIntArray) and 
            len(self.values) > 0 and len(other.values) > 0 and
            self.values._byteorder == other.values._byteorder == sys.byteorder and
            self.values._elemsize == other.values._elemsize and
            not use_internal
            ):
            try:
                return fast_intersection.intersection(
                    self.values, self.start, self.size, self.offset,
                    other.values, other.start, other.size, other.offset,
                )
            except NameError:
                pass

        result = []
        xiter = iter(self)
        yiter = iter(other)
        try:
            x = next(xiter)
            y = next(yiter)
            while True:
                if x < y:
                    x = next(xiter)
                elif x > y:
                    y = next(yiter)
                else:
                    result.append(x)
                    x = next(xiter)
                    y = next(yiter)
        except StopIteration:
            return result

    def difference_update(self, other:'IndexSet', use_internal:bool=False):
        self.values = self.difference(other, use_internal=use_internal)
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    def difference(self, other:'IndexSet', use_internal:bool=False) -> List[int]:
        """Take the difference between this set and another."""
        if (isinstance(self.values, DiskIntArray) and 
            isinstance(other.values, DiskIntArray) and 
            len(self.values) > 0 and len(other.values) > 0 and
            self.values._byteorder == other.values._byteorder == sys.byteorder and
            self.values._elemsize == other.values._elemsize and
            not use_internal
            ):
            try:
                return fast_intersection.difference(
                    self.values, self.start, self.size, self.offset,
                    other.values, other.start, other.size, other.offset,
                )
            except NameError:
                pass

        result = []
        xiter = iter(self)
        yiter = iter(other)
        try:
            x = next(xiter)
            y = next(yiter)
            while True:
                if x < y:
                    result.append(x)
                    x = next(xiter)
                elif x > y:
                    y = next(yiter)
                else:
                    x = next(xiter)
                    y = next(yiter)
        except StopIteration:
            return result

    def filter(self, check:Callable[[int],bool]):
        if isinstance(self.values, list):
            self._filter_values_in_place(self.values, check)
        else:
            self.values = [elem for elem in self if check(elem)]
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    @staticmethod
    def _filter_values_in_place(values:List[int], check:Callable[[int],bool]):
        filtered = 0
        for val in values:
            if check(val):
                values[filtered] = val
                filtered += 1
        del values[filtered:]

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

