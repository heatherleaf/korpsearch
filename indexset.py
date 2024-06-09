
import sys
import itertools
from pathlib import Path
from typing import Optional
from collections.abc import Iterator, Callable

from disk import DiskIntArray
from enum import Enum
from util import binsearch_lookup
import merge

fast_merge = None
try:
    import fast_merge  # type: ignore
except ModuleNotFoundError:
    print("Module 'fast_merge' not found. "
          "To install, run: 'python setup.py build_ext --inplace'. "
          "Defaulting to an internal implementation.\n", 
          file=sys.stderr)


################################################################################
## Different ways of merging sets: union, intersection, difference

class MergeType(Enum):
    """Types of merges that can be done."""
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2

    def which_to_take(self) -> tuple[bool, bool, bool]:
        """Compute take_first/take_second/take_common parameters (see comment in fast_merge.pyx)."""
        return {
            MergeType.UNION:        (True,  True,  True),
            MergeType.INTERSECTION: (False, False, True),
            MergeType.DIFFERENCE:   (True,  False, False)
        }[self]

    def max_merge_size(self, size1: int, size2: int) -> int:
        """Return the maximum size of the merge of two sets."""
        return {
            MergeType.UNION:        size1 + size2,
            MergeType.INTERSECTION: min(size1, size2),
            MergeType.DIFFERENCE:   size1,
        }[self]


################################################################################
## Index set

class IndexSet:
    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection / difference
    _min_size_difference_for_contains = 100

    start: int
    size: int
    offset: int
    values: DiskIntArray
    path: Optional[Path]

    def __init__(self, values: DiskIntArray, path: Optional[Path] = None, 
                 start: int = 0, size: int = -1, offset: int = 0) -> None:
        if size < 0:
            size = len(values) - start
        while size > 0 and values.array[start] < offset:
            start += 1
            size -= 1
        self.values = values
        self.path = path
        self.start = start
        self.offset = offset
        self.size = size

    @staticmethod
    def open(path: Path) -> 'IndexSet':
        arr = DiskIntArray(path)
        return IndexSet(arr, path)

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
        for val in self.values.array[self.start : self.start+self.size]:
            yield val - offset

    def __getitem__(self, i: int) -> int:
        if i < 0 or i >= self.size:
            raise IndexError("IndexSet index out of range")
        return self.values.array[self.start+i] - self.offset

    def __contains__(self, elem: int) -> bool:
        values = self.values.array
        offset = self.offset
        start = self.start
        end = start + self.size - 1
        return binsearch_lookup(start, end, elem, lambda i: values[i] - offset)

    def slice(self, start: int, end: int) -> Iterator[int]:
        if start < 0 or start >= self.size:
            raise IndexError("IndexSet index out of range")
        end = min(end, self.size)
        start += self.start
        end += self.start
        offset = self.offset
        for val in self.values.array[start : end]:
            yield val - offset

    def merge_update(self, other: 'IndexSet', resultpath: Optional[Path] = None, 
                     use_internal: bool = False, merge_type: MergeType = MergeType.INTERSECTION) -> str:
        # The returned string is ONLY for debugging purposes, and can safely be ignored

        take_first, take_second, take_common = merge_type.which_to_take()

        if take_common and (take_first != take_second or len(self) == 0 or len(other) == 0):
            if take_second or len(self) == 0:
                self.values = other.values
                self.path = other.path
                self.start = other.start
                self.size = other.size
                self.offset = other.offset
            return 'trivial'

        if not take_second and len(other) > len(self) * self._min_size_difference_for_contains:
            if take_common:
                self.filter_update(lambda x: x in other, resultpath)
            else:
                self.filter_update(lambda x: x not in other, resultpath)
            return 'lookup'

        result = self._init_result(resultpath, other, merge_type)
        merge_module = merge
        if not use_internal and fast_merge:
            merge_module = fast_merge
        final_size = merge_module.merge(  # type: ignore
            self.values.array, self.start, self.size, self.offset,
            other.values.array, other.start, other.size, other.offset,
            result.array, take_first, take_second, take_common,
        )
        self._finalise_result(result, final_size)  # type: ignore
        return 'internal' if merge_module is merge else 'external'

    def filter_update(self, check: Callable[[int],bool], resultpath: Optional[Path] = None) -> None:
        result = self._init_result(resultpath)
        i = 0
        for val in self:
            if check(val):
                result.array[i] = val
                i += 1
        self._finalise_result(result, i)

    def _init_result(self, resultpath: Optional[Path], other: Optional['IndexSet'] = None, 
                     merge_type: MergeType = MergeType.INTERSECTION) -> DiskIntArray:
        max_size = merge_type.max_merge_size(len(self), len(other)) if other else len(self)
        if not resultpath:
            return DiskIntArray.create(max_size, itemsize = self.values.array.itemsize)
        if self.path and DiskIntArray.getpath(self.path) == DiskIntArray.getpath(resultpath):
            return self.values
        if other and other.path: 
            assert DiskIntArray.getpath(other.path) != DiskIntArray.getpath(resultpath)
        return DiskIntArray.create(max_size, resultpath)

    def _finalise_result(self, result: DiskIntArray, size: int) -> None:
        assert 0 <= size <= len(result)
        if size < len(result):
            result.truncate(size)
        self.values = result
        self.path = result.path
        self.start = 0
        self.size = len(result)
        self.offset = 0

