"""General-purpose on-disk data structures."""

import os
import sys
import json
import math
from pathlib import Path
import mmap
from array import array
import itertools
from functools import total_ordering


def open_readonly_mmap(file):
    try:
        return mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
    except TypeError:
        # prot is not available on Windows
        return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)


def add_suffix(path, suffix):
    # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return Path(str(path) + suffix)


################################################################################
## On-disk arrays of numbers

CONFIG_SUFFIX = '.config.json'

def DiskIntArray(path):
    typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    for n, c in typecodes.items(): assert array(c).itemsize == n

    path = Path(path)
    file = open(path, 'rb')
    with open(add_suffix(path, CONFIG_SUFFIX)) as configfile:
        config = json.load(configfile)
    elemsize = config['elemsize']
    byteorder = config['byteorder']

    arr = open_readonly_mmap(file)
    if elemsize in typecodes and byteorder == sys.byteorder:
        return memoryview(arr).cast(typecodes[elemsize])
    else:
        return SlowDiskIntArray(arr, elemsize, byteorder)


class SlowDiskIntArray:
    def __init__(self, array, elemsize, byteorder):
        self._array = array
        self._elemsize = elemsize
        self._byteorder = byteorder
        self._length = len(array) // elemsize
        assert len(array) % elemsize == 0, f"Array length ({len(array)}) is not divisible by elemsize ({elemsize})"

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._slice(i))
        if not isinstance(i, int):
            raise TypeError("SlowDiskIntArray: invalid array index type")
        if i < 0 or i >= len(self):
            raise IndexError("SlowDiskIntArray: array index out of range")

        start = i * self._elemsize
        return int.from_bytes(self._array[start:start+self._elemsize], byteorder=self._byteorder)

    def _slice(self, slice):
        start, stop, step = slice.indices(len(self))
        array = self._array
        elemsize = self._elemsize
        byteorder = self._byteorder

        start *= elemsize
        stop *= elemsize
        step *= elemsize
        for i in range(start, stop, step):
            yield int.from_bytes(array[i:i+elemsize], byteorder=byteorder)

    def __iter__(self):
        return self._slice(slice(None))


class DiskIntArrayBuilder:
    def __init__(self, path, max_value=None, byteorder=None, use_memoryview=False):
        if max_value is None: max_value = 2**32-1
        if byteorder is None: byteorder = sys.byteorder
        self._path = Path(path)
        self._byteorder = byteorder

        self._elemsize = self._min_bytes_to_store_values(max_value)
        if use_memoryview:
            if byteorder != sys.byteorder:
                raise ValueError(f"DiskIntArrayBuilder: memoryview requires byteorder to be '{sys.byteorder}'")
            if self._elemsize == 3: self._elemsize = 4
            if self._elemsize > 4 and self._elemsize <= 8: self._elemsize = 8
            if self._elemsize > 8:
                raise RuntimeError('DiskIntArrayBuilder: memoryview does not support self._elemsize > 8')

        with open(str(path) + CONFIG_SUFFIX, 'w') as configfile:
            json.dump({
                'elemsize': self._elemsize,
                'byteorder': self._byteorder,
            }, configfile)
        self._file = open(path, 'wb')

    def append(self, value):
        self._file.write(value.to_bytes(self._elemsize, byteorder=self._byteorder))

    def __setitem__(self, k, value):
        self._file.seek(k * self._elemsize)
        self.append(value)
        self._file.seek(0, os.SEEK_END)

    def __len__(self):
        return self._file.tell() // self._elemsize

    def close(self):
        self._file.close()
        self._file = None

    @staticmethod
    def _min_bytes_to_store_values(max_value):
        return math.ceil(math.log(max_value + 1, 2) / 8)

    @staticmethod
    def build(path, values, max_value=None, byteorder=None, use_memoryview=False):
        if max_value is None:
            values = list(values)
            max_value = max(values)

        builder = DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview)
        for value in values: 
            builder.append(value)
        builder.close()
        return DiskIntArray(path)


################################################################################
## String interning

STARTS_SUFFIX  = '.starts'

class StringCollection:
    def __init__(self, path):
        path = Path(path)
        stringsfile = open(path, 'rb')
        self._path = path
        self._strings = open_readonly_mmap(stringsfile)
        self._starts = DiskIntArray(add_suffix(path, STARTS_SUFFIX))
        self._intern = None

    def __len__(self):
        return len(self._starts)-1

    def close(self):
        self._stringsfile.close()
        self._starts.close()
        self._stringsfile = self._starts = None

    def from_index(self, index):
        return InternedString(self, index)

    def preload(self):
        if self._intern is None:
            self._intern = {}
            for i in range(len(self)):
                self._intern[bytes(self.from_index(i))] = i

    def intern(self, string):
        if isinstance(string, InternedString):
            assert string._db is self
            return string

        if self._intern is not None:
            return self.from_index(self._intern[string])

        lo = 0
        hi = len(self)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            here = bytes(self.from_index(mid))
            if string < here:
                hi = mid-1
            elif string > here:
                lo = mid+1
            else:
                return self.from_index(mid)
        raise KeyError(f"StringCollection: string '{string}' not found in database")


class StringCollectionBuilder:
    @staticmethod
    def build(path, strings):
        stringlist = sorted(set(strings))

        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)

        stringlist.insert(0, '')  # this is to emulate the 'initial' keyword in accumulate, which was introduced in Python 3.8
        DiskIntArrayBuilder.build(
            add_suffix(path, STARTS_SUFFIX),
            itertools.accumulate((len(s) for s in stringlist)),
            use_memoryview = True
        )

        return StringCollection(path)


@total_ordering
class InternedString:
    __slots__ = ['_db', 'index']

    def __init__(self, db, index):
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "index", index)

    def __bytes__(self):
        start = self._db._starts[self.index]
        nextstart = self._db._starts[self.index+1]
        return self._db._strings[start:nextstart]

    def __str__(self):
        return str(bytes(self))

    def __repr__(self):
        return f"InternedString({self})"

    def __eq__(self, other):
        if isinstance(other, InternedString) and self._db is other._db:
            return self.index == other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __lt__(self, other):
        if isinstance(other, InternedString) and self._db is other._db:
            return self.index < other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __hash__(self):
        return hash(self.index)

    def __setattr__(self, _field, _value):
        raise TypeError("InternedString is read-only")

    def __delattr__(self, _field):
        raise TypeError("InternedString is read-only")


################################################################################
## On-disk arrays of interned strings

STRINGS_SUFFIX = '.strings'

class DiskStringArray:
    def __init__(self, path):
        path = Path(path)
        self._array = DiskIntArray(path)
        self._strings = StringCollection(add_suffix(path, STRINGS_SUFFIX))

    def raw(self):
        return self._array

    def intern(self, x):
        return self._strings.intern(x)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._slice(i))

        if not isinstance(i, int):
            raise TypeError("invalid array index type")

        return self._strings.from_index(self._array[i])

    def _slice(self, slice):
        return map(self._strings.from_index, self._array[slice])

    def __iter__(self):
        yield from self._slice(slice(None))


class DiskStringArrayBuilder:
    def __init__(self, path, strings, use_memoryview=False):
        self._path = Path(path)
        strings_path = add_suffix(path, STRINGS_SUFFIX)
        StringCollectionBuilder.build(strings_path, strings)
        self._strings = StringCollection(strings_path)
        self._strings.preload()
        self._builder = DiskIntArrayBuilder(path, max_value=len(self._strings)-1, use_memoryview=use_memoryview)

    def append(self, value):
        self._builder.append(self._strings.intern(value).index)

    def close(self):
        self._builder.close()
        self._builder = None

    @staticmethod
    def build(path, values, strings=None, use_mmap=False):
        if strings is None:
            values = strings = list(values)

        builder = DiskStringArrayBuilder(path, strings, use_mmap)
        for value in values:
            builder.append(value)
        builder.close()

        return DiskStringArray(path)
