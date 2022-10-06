
import sys
import itertools
from pathlib import Path
from typing import List, Iterator, Union, Callable, Optional

from disk import DiskIntArray, LowlevelIntArray, DiskIntArrayBuilder

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
IndexSetValuesBuilder = Union[DiskIntArray, DiskIntArrayBuilder, List[int]]

class IndexSet:
    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection / difference
    _min_size_difference_for_contains = 100

    start : int
    size : int
    offset : int
    values : IndexSetValuesType
    path : Optional[Path]

    def __init__(self, values:IndexSetValuesType, start:int=0, size:int=-1, offset:int=0):
        self.values = values
        self.path = values.path if isinstance(values, DiskIntArray) else None
        self.start = start
        self.offset = offset
        self.size = size
        if size < 0:
            self.size = len(values) - start
        while self.values[self.start] < self.offset:
            self.start += 1
            self.size -= 1
        assert self.size > 0

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        MAX = 5
        s = ', '.join(str(n) for n in itertools.islice(self, MAX))
        if len(self) > MAX:
            s += f", ... (N={len(self)})"
        if self.path:
            s += f" @ {self.path}"
        return "{" + s + "}"

    def __iter__(self) -> Iterator[int]:
        offset = self.offset
        for val in self.values[self.start:self.start+self.size]:
            yield val - offset


    def intersection_update(self, other:'IndexSet', resultpath:Optional[Path]=None, use_internal:bool=False, difference:bool=False) -> str:
        # The returned string is ONLY for debugging purposes, and can safely be ignored
        if len(other) > len(self) * self._min_size_difference_for_contains:
            result = self._init_result(resultpath, other)
            self._intersection_lookup(other, result, difference)
            self._finalise_result(result)
            return 'lookup'

        if not use_internal and not resultpath:
            result = self._intersection_external(other, difference)
            if result is not None:
                self._finalise_result(result)
                return 'external'

        result = self._init_result(resultpath, other)
        self._intersection_internal(other, result, difference)
        self._finalise_result(result)
        return 'internal'

    def _init_result(self, resultpath:Optional[Path], other:Optional['IndexSet']=None) -> IndexSetValuesBuilder:
        if not resultpath:
            return []
        elif self.path and DiskIntArray.getpath(resultpath) == DiskIntArray.getpath(self.path):
            assert isinstance(self.values, DiskIntArray) and not isinstance(self.values, LowlevelIntArray)
            self.values.reset_append()
            return self.values
        else:
            if other and other.path: 
                assert DiskIntArray.getpath(resultpath) != DiskIntArray.getpath(other.path)
            return DiskIntArrayBuilder(resultpath)

    def _finalise_result(self, result:IndexSetValuesBuilder):
        path = None
        if isinstance(result, DiskIntArray) and not isinstance(result, LowlevelIntArray):
            result.truncate_append()
            path = result.path
        elif isinstance(result, DiskIntArrayBuilder):
            path = result.path
            result.close()
            result = DiskIntArray(path)
        self.values = result
        self.path = path
        self.start = 0
        self.size = len(self.values)
        self.offset = 0

    def _intersection_lookup(self, other:'IndexSet', result:IndexSetValuesBuilder, difference:bool):
        # Complexity: O(self * log(other))
        if difference:
            for elem in self:
                if elem not in other:
                    result.append(elem)
        else: # intersection
            for elem in self:
                if elem in other:
                    result.append(elem)

    def _intersection_external(self, other:'IndexSet', difference:bool) -> Optional[IndexSetValuesType]:
        # Complexity: O(self + other)
        if (isinstance(self.values, DiskIntArray) and 
            isinstance(other.values, DiskIntArray) and 
            len(self) > 0 and len(other) > 0 and
            self.values._byteorder == other.values._byteorder == sys.byteorder and
            self.values._elemsize == other.values._elemsize
        ):
            try:
                return fast_intersection.intersection(
                    self.values, self.start, self.size, self.offset,
                    other.values, other.start, other.size, other.offset, difference,
                )
            except NameError:
                pass

    def _intersection_internal(self, other:'IndexSet', result:IndexSetValuesBuilder, difference:bool):
        # Complexity: O(self + other)
        xiter = iter(self)
        yiter = iter(other)
        try:
            x = next(xiter)
            y = next(yiter)
            while True:
                if x < y:
                    if difference: result.append(x)
                    x = next(xiter)
                elif x > y:
                    y = next(yiter)
                else:
                    if not difference: result.append(x)
                    x = next(xiter)
                    y = next(yiter)
        except StopIteration:
            return


    def filter_update(self, check:Callable[[int],bool], resultpath:Optional[Path]=None):
        result = self._init_result(resultpath)
        for val in self.values:
            if check(val):
                result.append(val)
        self._finalise_result(result)

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

