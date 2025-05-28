
from enum import Enum
from dataclasses import dataclass
from collections.abc import Iterator, Callable

from pyroaring import BitMap


################################################################################
## Different ways of merging sets: union, intersection, difference

class MergeType(Enum):
    """Types of merges that can be done."""
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2


################################################################################
## Index set

@dataclass
class IndexSet:
    values: BitMap

    def __len__(self) -> int:
        return len(self)

    def __str__(self) -> str:
        MAX = 3
        if len(self) <= MAX:
            return str(set(self))
        else:
            return f"{{{self.values.min()}...{self.values.max()} (N={len(self)})}}"

    def __iter__(self) -> Iterator[int]:
        yield from self.values

    def __getitem__(self, i: int) -> int:
        return self.values[i]

    def __contains__(self, elem: int) -> bool:
        return elem in self.values

    def slice(self, start: int, end: int) -> Iterator[int]:
        if start < 0 or start >= len(self):
            raise IndexError("IndexSet index out of range")
        yield from self.values[start:end]

    def merge_update(self, other: 'IndexSet', merge_type: MergeType = MergeType.INTERSECTION) -> None:
        if merge_type == MergeType.INTERSECTION:
            self.values.intersection_update(other.values)
        elif merge_type == MergeType.UNION:
            self.values.update(other.values)
        elif merge_type == MergeType.DIFFERENCE:
            self.values.difference_update(other.values)

    def filter_update(self, check: Callable[[int],bool]) -> None:
        self.values = BitMap(val for val in self.values if check(val))

