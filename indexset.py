
import sys
import itertools
from pathlib import Path
from typing import Union, Optional
from collections.abc import Iterator, Callable

from disk import DiskIntArray, LowlevelIntArray, DiskIntArrayBuilder
from enum import Enum

try:
    import platform
    if platform.python_implementation() == 'CPython':
        # fast_merge is NOT faster in PyPy!
        import fast_merge  # type: ignore
except ModuleNotFoundError:
    print("Module 'fast_merge' not found. "
          "To install, run: 'python setup.py build_ext --inplace'. "
          "Defaulting to an internal implementation.\n", 
          file=sys.stderr)


################################################################################
## Index set

IndexSetValuesType = Union[DiskIntArray, list[int]]
IndexSetValuesBuilder = Union[DiskIntArray, DiskIntArrayBuilder, list[int]]

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


class IndexSet:
    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection / difference
    _min_size_difference_for_contains = 100

    start: int
    size: int
    offset: int
    values: IndexSetValuesType
    path: Optional[Path]

    def __init__(self, values: IndexSetValuesType, start: int = 0, size: int = -1, offset: int = 0) -> None:
        if size < 0:
            size = len(values) - start
        while size > 0 and values[start] < offset:
            start += 1
            size -= 1
        self.values = values
        self.path = values.path if isinstance(values, DiskIntArray) else None
        self.start = start
        self.offset = offset
        self.size = size

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
        for val in self.values[self.start : self.start+self.size]:
            yield val - offset

    def __getitem__(self, i: int) -> int:
        if i < 0 or i >= self.size:
            raise IndexError("IndexSet index out of range")
        return self.values[self.start+i] - self.offset

    def slice(self, start: int, end: int) -> Iterator[int]:
        if start < 0 or start >= self.size:
            raise IndexError("IndexSet index out of range")
        end = min(end, self.size)
        start += self.start
        end += self.start
        offset = self.offset
        for val in self.values[start : end]:
            yield val - offset

    def merge_update(self, other: 'IndexSet', resultpath: Optional[Path] = None, 
                     use_internal: bool = False, merge_type: MergeType = MergeType.INTERSECTION) -> str:
        # The returned string is ONLY for debugging purposes, and can safely be ignored
        if len(other) > len(self) * self._min_size_difference_for_contains and merge_type != MergeType.UNION:
            lokup_result = self._init_result(resultpath, other)
            if merge_type == MergeType.INTERSECTION:
                self._intersection_lookup(other, lokup_result)
            else:
                self._difference_lookup(other, lokup_result)
            self._finalise_result(lokup_result)
            return 'lookup'

        if not use_internal and not resultpath:
            external_result = self._merge_external(other, merge_type)
            if external_result is not None:
                self._finalise_result(external_result)
                return 'external'

        internal_result = self._init_result(resultpath, other)
        self._merge_internal(other, internal_result, merge_type)
        self._finalise_result(internal_result)
        return 'internal'

    def _init_result(self, resultpath: Optional[Path], other: Optional['IndexSet'] = None) -> IndexSetValuesBuilder:
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

    def _finalise_result(self, result: IndexSetValuesBuilder) -> None:
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

    def _intersection_lookup(self, other: 'IndexSet', result: IndexSetValuesBuilder) -> None:
        # Complexity: O(self * log(other))
        for elem in self:
            if elem in other:
                result.append(elem)

    def _difference_lookup(self, other: 'IndexSet', result: IndexSetValuesBuilder) -> None:
        for elem in self:
            if elem not in other:
                result.append(elem)

    def _merge_external(self, other: 'IndexSet', merge_type: MergeType) -> Optional[IndexSetValuesType]:
        # Complexity: O(self + other)
        if (isinstance(self.values, DiskIntArray) and 
            isinstance(other.values, DiskIntArray) and 
            len(self) > 0 and len(other) > 0 and
            self.values._byteorder == other.values._byteorder == sys.byteorder and  # type: ignore
            self.values._elemsize == other.values._elemsize  # type: ignore
        ):
            take_first, take_second, take_common = merge_type.which_to_take()
            try:
                return fast_merge.merge(  # type: ignore
                    self.values, self.start, self.size, self.offset,
                    other.values, other.start, other.size, other.offset,
                    take_first, take_second, take_common
                )
            except NameError:
                pass
        return None

    def _merge_internal(self, other: 'IndexSet', result: IndexSetValuesBuilder, merge_type: MergeType) -> None:
        # Complexity: O(self + other)
        take_first, take_second, take_common = merge_type.which_to_take()
        xiter = iter(self)
        yiter = iter(other)
        x = next(xiter, None)
        y = next(yiter, None)
        while x is not None and y is not None:
            if x < y:
                if take_first: result.append(x)
                x = next(xiter, None)
            elif x > y:
                if take_second: result.append(y)
                y = next(yiter, None)
            else:
                if take_common: result.append(x)
                x = next(xiter, None)
                y = next(yiter, None)

        if take_first:
            if x is not None:
                result.append(x)
            result.extend(xiter)

        if take_second:
            if y is not None:
                result.append(y)
            result.extend(yiter)


    def filter_update(self, check: Callable[[int],bool], resultpath: Optional[Path] = None) -> None:
        result = self._init_result(resultpath)
        for val in self:
            if check(val):
                result.append(val)
        self._finalise_result(result)

    def __contains__(self, elem: int) -> bool:
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

