"""General-purpose on-disk data structures."""

import sys
import json
from pathlib import Path
from mmap import mmap
import itertools
from typing import Optional, Union, Any, NewType
from collections.abc import Iterator, Iterable

from util import add_suffix, get_integer_size, get_typecode, binsearch, binsearch_range


################################################################################
## On-disk arrays of numbers

class DiskIntArray:
    array_suffix = '.ia'
    config_suffix = '.cfg'
    default_itemsize = 4

    array: memoryview
    path: Optional[Path] = None

    def __init__(self, source: Union[Path, mmap, bytearray], itemsize: int = default_itemsize) -> None:
        assert not isinstance(source, bytes), "bytes is not mutable - use bytearray instead"
        if isinstance(source, Path):
            with open(self.getconfig(source)) as configfile:
                config = json.load(configfile)
            assert config['byteorder'] == sys.byteorder, f"Cannot handle byteorder {config['byteorder']}"
            itemsize = config['itemsize']
            self.path = self.getpath(source)
            with open(self.path, 'r+b') as file:
                try:
                    source = mmap(file.fileno(), 0)
                except ValueError:  # "cannot mmap an empty file"
                    source = bytearray(0)
        self.array = memoryview(source).cast(get_typecode(itemsize))

    def __len__(self) -> int:
        return len(self.array)

    def __enter__(self) -> memoryview:
        return self.array

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        obj = self.array.obj
        self.array.release()
        if isinstance(obj, mmap):
            obj.close()

    def truncate(self, newsize: int) -> None:
        obj = self.array.obj
        itemsize = self.array.itemsize
        self.array.release()
        if isinstance(obj, bytearray):
            assert self.path is None
            del obj[newsize * itemsize:]
            typecode = get_typecode(itemsize)
            self.array = memoryview(obj).cast(typecode)
        elif isinstance(obj, mmap):
            assert self.path
            obj.close()
            with open(self.path, 'r+b') as file:
                file.truncate(newsize * itemsize)
                try:
                    obj = mmap(file.fileno(), 0)
                except ValueError:  # "cannot mmap an empty file"
                    obj = bytearray(0)
        else:
            raise ValueError(f"Cannot handle memoryview object {obj}")
        self.array = memoryview(obj).cast(get_typecode(itemsize))

    @staticmethod
    def create(size: int, path: Optional[Path] = None, max_value: int = 0, itemsize: int = 0) -> 'DiskIntArray':
        assert not (max_value and itemsize), "Only one of 'max_value' and 'itemsize' should be provided."
        if max_value > 0: 
            itemsize = get_integer_size(max_value)
        if not itemsize:
            itemsize = DiskIntArray.default_itemsize
        if path:
            with open(DiskIntArray.getconfig(path), 'w') as configfile:
                json.dump({
                    'itemsize': itemsize,
                    'byteorder': sys.byteorder,
                }, configfile)
            with open(DiskIntArray.getpath(path), 'wb') as file:
                file.truncate(size * itemsize)
            return DiskIntArray(path)
        else:
            data = bytearray(size * itemsize)
            return DiskIntArray(data, itemsize)

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.array_suffix)

    @classmethod
    def getconfig(cls, path: Path) -> Path:
        return add_suffix(cls.getpath(path), cls.config_suffix)


################################################################################
## String interning

InternedString = NewType('InternedString', int)
InternedRange = tuple[InternedString, InternedString]


class StringCollection:
    strings_suffix = '.strings'

    strings: mmap
    starts: DiskIntArray

    _intern: dict[bytes, InternedString]

    def __init__(self, path: Path, preload: bool = False) -> None:
        path = self.getpath(path)
        with open(path, 'r+b') as file:
            self.strings = mmap(file.fileno(), 0)
        self.starts = DiskIntArray(path)
        assert self.starts.array[0]+1 == self.starts.array[1]
        self._intern = {}
        if preload:
            self.preload()

    def __len__(self) -> int:
        return len(self.starts) - 1

    def from_index(self, index: int) -> bytes:
        arr = self.starts.array
        start, nextstart = arr[index], arr[index + 1]
        return self.strings[start : nextstart-1]

    def preload(self) -> None:
        if not self._intern:
            self._intern = {}
            for i in range(len(self)):
                self._intern[self.from_index(i)] = InternedString(i)

    def intern(self, string: bytes) -> InternedString:
        if self._intern:
            return self._intern[string]
        else:
            return InternedString(binsearch(0, len(self)-1, string, lambda i: self.from_index(i)))

    def interned_range(self, string: bytes) -> InternedRange:
        n = len(string)
        start, end = binsearch_range(0, len(self)-1, string, lambda i: self.from_index(i)[:n])
        return (InternedString(start), InternedString(end))

    def __enter__(self) -> 'StringCollection':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.strings.close()
        self.starts.close()

    def sanity_check(self) -> None:
        starts = self.starts.array
        assert starts[0] == 0 and starts[1] == 1
        old = b''
        for start, end in zip(starts[1:], starts[2:]):
            assert start < end, f"StringCollection position error: {start} >= {end}"
            new = self.strings[start : end]
            assert old < new, f"StringCollection order error: {old!r} >= {new!r}"
            old = new


    @staticmethod
    def build(path: Path, strings: Iterable[bytes]) -> None:
        stringset = set(strings)
        stringset.add(b'')
        stringlist = sorted(stringset)
        assert stringlist[0] == b''

        path = StringCollection.getpath(path)
        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)
                stringsfile.write(b'\n')

        starts = list(itertools.accumulate((len(s)+1 for s in stringlist), initial=0))
        with DiskIntArray.create(len(starts), path, max_value=starts[-1]) as arr:
            for i, start in enumerate(starts):
                arr[i] = start
            assert arr[0] == 0 and arr[1] == 1

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.strings_suffix)


################################################################################
## On-disk arrays of interned strings

class DiskStringArray:
    _strings: StringCollection
    _array: DiskIntArray

    def __init__(self, path: Path, preload: bool = False) -> None:
        self._array = DiskIntArray(path)
        self._strings = StringCollection(path, preload)

    def raw(self) -> memoryview:
        return self._array.array

    def intern(self, x: bytes) -> InternedString:
        return self._strings.intern(x)

    def interned_range(self, x: bytes) -> InternedRange:
        return self._strings.interned_range(x)

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, i: int) -> InternedString:
        return InternedString(self._array.array[i])

    def interned_bytes(self, s: InternedString) -> bytes:
        return self._strings.from_index(s)

    def interned_string(self, s: InternedString) -> str:
        return self.interned_bytes(s).decode()

    def __setitem__(self, i: int, value: InternedString) -> None:
        self._array.array[i] = value

    def __iter__(self) -> Iterator[InternedString]:
        for i in self._array.array:
            yield InternedString(i)

    def __enter__(self) -> 'DiskStringArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._array.close()
        self._strings.close()

    def sanity_check(self) -> None:
        self._strings.sanity_check()


    @staticmethod
    def create(path: Path, strings: Iterable[bytes], max_size: int) -> 'DiskStringArray':
        StringCollection.build(path, strings)
        collection = StringCollection(path, preload=True)
        DiskIntArray.create(max_size, path, max_value = len(collection)-1)
        return DiskStringArray(path, preload=True)

