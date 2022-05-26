
import itertools
from pathlib import Path
from disk import DiskIntArray, DiskIntArrayType, InternedString
from corpus import Corpus
from typing import Tuple, List, Iterator, Union, Callable

################################################################################
## Templates and instances

class Template:
    def __init__(self, *feature_positions:Tuple[bytes,int]):
        self._feature_positions : Tuple[Tuple[bytes,int],...] = feature_positions

    def __bytes__(self) -> bytes:
        return b'-'.join(feat + str(pos).encode() for feat, pos in self._feature_positions)

    def __str__(self) -> str:
        return '-'.join(feat.decode() + str(pos) for feat, pos in self._feature_positions)

    def __iter__(self) -> Iterator[Tuple[bytes,int]]:
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
        self.basedir = corpus.path().with_suffix(self.dir_suffix)
        self.corpus = corpus
        self.template = template
        basefile : Path = self.basefile()

        self.keypaths = [basefile.with_suffix(f'.{feature.decode()}{pos}') for feature, pos in template]
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
        return self.basedir / str(self.template)

    # def close(self):
    #     for keyarray in self._keys: self._close(keyarray)
    #     self._close(self._index)
    #     self._close(self._sets)

    #     self._keys = []
    #     self._index = None
    #     self._sets = None

    # def _close(self, file : DiskIntArrayType):
    #     if hasattr(file, 'close'):
    #         file.close()

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

    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection
    _min_size_difference : int = 1000

    def intersection_update(self, other:'IndexSet'):
        # We assume that self is smaller than other!
        if len(other) > len(self) * self._min_size_difference:
            # O(self * log(other))
            self.values = [elem for elem in self if elem in other]

        elif isinstance(self.values, set):
            # O(self + other)
            self.values.intersection_update(other)

        else:
            # O(self + other)
            result : List[int] = []
            self_iter : Iterator[int] = iter(sorted(self))
            other_iter : Iterator[int] = iter(other)
            self_val : int = next(self_iter)
            other_val : int = next(other_iter)
            while True:
                try:
                    if self_val == other_val:
                        result.append(self_val)
                        self_val = next(self_iter)
                        other_val = next(other_iter)
                    elif self_val < other_val:
                        self_val = next(self_iter)
                    else: # selfval > otherval
                        other_val = next(other_iter)
                except StopIteration:
                    break
            self.values = result

        if not self.values:
            raise ValueError("Empty intersection")
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

