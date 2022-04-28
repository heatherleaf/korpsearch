"""General-purpose on-disk data structures."""

import os
import math
from pathlib import Path
import mmap
from array import array
import itertools

################################################################################
## On-disk arrays of numbers

ENDIANNESS = 'little'

class DiskIntArray:
    _typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    for n, c in _typecodes.items(): assert array(c).itemsize == n

    _headersize = 8

    def __init__(self, path):
        path = Path(path)
        self._file = open(path, 'rb')
        self._elemsize = int.from_bytes(self._file.read(self._headersize), byteorder=ENDIANNESS)

        self._file.seek(0, os.SEEK_END)
        self._length = (self._file.tell() - self._headersize) // self._elemsize

        if self._elemsize in self._typecodes:
            bytes = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
            self._array = memoryview(bytes).cast(self._typecodes[self._elemsize])
            self._headercount = self._headersize // self._elemsize
        else:
            self._array = None

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._slice(i))

        if not isinstance(i, int):
            raise TypeError("invalid array index type")

        if i < 0 or i >= len(self):
            raise IndexError("array index out of range")

        if self._array is None:
            self._file.seek(self._headersize + i * self._elemsize)
            return int.from_bytes(self._file.read(self._elemsize), byteorder=ENDIANNESS)
        else:
            return self._array[self._headercount + i]

    def _slice(self, slice):
        start, stop, step = slice.indices(len(self))
        if step != 1: raise IndexError("only slices with step 1 supported")

        if self._array is None:
            for i in range(start, stop):
                yield self[i]
        else:
            start += self._headercount
            stop += self._headercount
            yield from self._array[start:stop]

    def __iter__(self):
        if self._array is None:
            self._array.seek(self._headersize)
            for _ in range(self._length):
                yield int.from_bytes(self._array.read(self._elemsize), byteorder=ENDIANNESS)
        else:
            yield from self._array[self._headercount:]

    @staticmethod
    def build(path, values, max_value = None, use_mmap=False):
        if max_value is None:
            values = list(values)
            max_value = max(values)

        builder = DiskIntArrayBuilder(path, max_value, use_mmap)
        for value in values: builder.append(value)
        builder.close()
        return DiskIntArray(path)

class DiskIntArrayBuilder:
    def __init__(self, path, max_value, use_mmap=False):
        self._path = Path(path)
        self._max_value = max_value

        self._elem_size = self._min_bytes_to_store_values(max_value)
        if use_mmap:
            if self._elem_size == 3: self._elem_size = 4
            if self._elem_size > 4 and self._elem_size <= 8: self._elem_size = 8
            if self._elem_size > 8:
                raise RuntimeError('DiskIntArray: use_mmap=True not supported with self._elem_size > 8')

        self._file = open(path, 'wb')
        self._file.write(self._elem_size.to_bytes(DiskIntArray._headersize, byteorder=ENDIANNESS))

    def append(self, value):
        self._file.write(value.to_bytes(self._elem_size, byteorder=ENDIANNESS))

    def close(self):
        self._file.close()
        self._file = None

    def _min_bytes_to_store_values(self, max_value):
        return math.ceil(math.log(max_value + 1, 2) / 8)

################################################################################
## String interning

class StringCollection:
    def __init__(self, path):
        path = Path(path)
        stringsfile = open(path, 'rb')
        self._strings = mmap.mmap(stringsfile.fileno(), 0, prot=mmap.PROT_READ)
        self._starts = DiskIntArray(path.with_suffix(path.suffix + '.starts'))
        self._lengths = DiskIntArray(path.with_suffix(path.suffix + '.lengths'))
        self._intern = None

    def __len__(self):
        return len(self._starts)

    def unintern(self, index):
        start = self._starts[index]
        length = self._lengths[index]
        return self._strings[start:start+length]

    def fast_intern(self, string):
        if self._intern is None:
            self._intern = {}
            for i in range(len(self)):
                self._intern[self.unintern(i)] = i

        return self._intern[string]

    def intern(self, string):
        lo = 0
        hi = len(self)-1
        while lo <= hi:
            mid = (lo+hi)//2
            here = self.unintern(mid)
            if string < here:
                hi = mid-1
            elif string > here:
                lo = mid+1
            else:
                return mid
        assert False, "string not found in database"

    @staticmethod
    def build(path, strings):
        stringset = set()
        for i, word in enumerate(strings):
            stringset.add(word)

        stringlist = list(stringset)
        stringlist.sort()

        with open(path.with_suffix('.strings'), 'wb') as stringsfile:
            for string in stringlist: stringsfile.write(string)
            size = stringsfile.tell()

        DiskIntArray.build(path.with_suffix('.strings.starts'),
            itertools.accumulate((len(s) for s in stringlist[:-1]), initial=0),
            max_value = size),

        DiskIntArray.build(path.with_suffix('.strings.lengths'),
            (len(s) for s in stringlist),
            max_value = size)

        return StringCollection(path)

################################################################################
## On-disk arrays of interned strings

class DiskStringArray:
    def __init__(self, path):
        path = Path(path)
        self._array = DiskIntArray(path)
        self._strings = StringCollection(path.with_suffix(path.suffix + '.strings'))

    def raw(self):
        return self._array

    def fast_intern(self, x):
        return self._strings.fast_intern(x)

    def intern(self, x):
        return self._strings.intern(x)

    def unintern(self, x):
        return self._strings.unintern(x)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._slice(i))

        if not isinstance(i, int):
            raise TypeError("invalid array index type")

        return self._strings.unintern(self._array[i])

    def _slice(self, slice):
        for str in self._array[slice]:
            yield self._strings.unintern(str)

    def __iter__(self):
        yield from self._slice(slice(None))

    @staticmethod
    def build(path, values, strings=None, use_mmap=False):
        if strings is None:
            values = strings = list(values)

        builder = DiskStringArrayBuilder(path, strings, use_mmap)
        for value in values:
            builder.append(value)
        builder.close()

        return DiskStringArray(path)

class DiskStringArrayBuilder:
    def __init__(self, path, strings, use_mmap=False):
        self._path = Path(path)
        strings_path = path.with_suffix(path.suffix + '.strings')
        StringCollection.build(strings_path, strings)
        self._strings = StringCollection(strings_path)
        self._builder = DiskIntArrayBuilder(path, len(self._strings)-1, use_mmap)

    def append(self, value):
        self._builder.append(self._strings.fast_intern(value))

    def close(self):
        self._builder.close()
        self._builder = None
