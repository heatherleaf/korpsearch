
import itertools
from pathlib import Path
from typing import Tuple, List, Iterator, Union, Callable
import sys

from disk import DiskIntArray, DiskIntArrayType, InternedString
from corpus import Corpus

try:
    from fast_intersection import intersection  # type: ignore
except ModuleNotFoundError:
    print("""
Module 'fast_intersection' not found. To install, run: 'python setup.py build_ext --inplace'.
Using a slow internal implementation instead.
""", file=sys.stderr)

    def intersection(arr1:DiskIntArrayType, start1:int, length1:int, 
                     arr2:DiskIntArrayType, start2:int, length2:int) -> List[int]:
        """Take the intersection of two sorted arrays."""
        result = []
        k1, k2 = 0, 0
        x1, x2 = arr1[start1], arr2[start2]
        while k1 < length1 and k2 < length2:
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
        return result


################################################################################
## Templates and instances

class Template:
    def __init__(self, *feature_positions:Tuple[str,int]):
        assert len(feature_positions) > 0
        assert feature_positions[0][-1] == 0
        self._feature_positions : Tuple[Tuple[str,int],...] = feature_positions

    def __bytes__(self) -> bytes:
        return str(self).encode()

    def __str__(self) -> str:
        return '+'.join(feat + str(pos) for feat, pos in self._feature_positions)

    def __iter__(self) -> Iterator[Tuple[str,int]]:
        yield from self._feature_positions

    def __len__(self) -> int:
        return len(self._feature_positions)


class Instance:
    def __init__(self, *values : InternedString):
        self._values : Tuple[InternedString,...] = values

    def values(self) -> Tuple[InternedString,...]:
        return self._values

    def __bytes__(self) -> bytes:
        return b' '.join(map(bytes, self._values))

    def __str__(self) -> str:
        return ' '.join(map(str, self._values))

    def __iter__(self) -> Iterator[InternedString]:
        yield from self._values

    def __len__(self) -> int:
        return len(self._values)


################################################################################
## Inverted sentence index
## Implemented as a sorted array of interned strings

class Index:
    dir_suffix : str = '.indexes'

    basedir : Path
    corpus : Corpus
    template : Template

    keys : List[DiskIntArrayType]
    index : DiskIntArrayType
    sets : DiskIntArrayType

    keypaths : List[Path]
    indexpath : Path
    setspath : Path

    def __init__(self, corpus:Corpus, template:Template, mode:str='r'):
        assert mode in "rw"
        assert isinstance(template, Template)
        self.basedir = corpus.path.with_suffix(self.dir_suffix)
        self.corpus = corpus
        self.template = template

        basefile : Path = self.basefile()
        basefile.parent.mkdir(exist_ok=True)
        self.keypaths = [basefile.with_suffix(f'.key:{feature}{pos}') for feature, pos in template]
        self.indexpath = basefile.with_suffix('.index')
        self.setspath = basefile.with_suffix('.sets')

        if mode == 'r':
            self.keys = [DiskIntArray(path) for path in self.keypaths]
            self.index = DiskIntArray(self.indexpath)
            self.sets = DiskIntArray(self.setspath)

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self) -> int:
        return len(self.index)

    def basefile(self) -> Path:
        return self.basedir / str(self.template) / str(self.template)

    def search(self, instance:Instance) -> 'IndexSet':
        set_start : int = self._lookup_instance(instance)
        return IndexSet(self.sets, set_start)

    def _lookup_instance(self, instance:Instance) -> int:
        # binary search
        instance_key : Tuple[int,...] = tuple(s.index for s in instance)
        start : int = 0; end : int = len(self)-1
        while start <= end:
            mid : int = (start + end) // 2
            key : Tuple[int,...] = tuple(keyarray[mid] for keyarray in self.keys)
            if key == instance_key:
                return self.index[mid]
            elif key < instance_key:
                start = mid + 1
            else:
                end = mid - 1
        raise KeyError(f'Instance "{instance}" not found')


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
        if isinstance(values, list):
            self.size = size if size >= 0 else len(values) - start
        else:
            assert size == -1
            self.size = values[start]
            self.start += 1

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

        self.values = intersection(
            self.values, self.start, self.size,
            other.values, other.start, other.size,
        )
        self.start = 0
        self.size = len(self.values)

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

