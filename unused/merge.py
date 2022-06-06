from _merge import ffi, lib
from disk import SlowDiskIntArray
import sys

def to_bytes_and_size(array):
    if isinstance(array, memoryview):
        return array.obj, array.itemsize
    elif isinstance(array, SlowDiskIntArray):
        assert array._byteorder == sys.byteorder
        return array._array, array._elemsize
    else:
        assert False, "argument to to_bytes_and_size has unknown type"

def intersection(in1, start1, size1, in2, start2, size2):
    in1, elemsize = to_bytes_and_size(in1)
    in2, elemsize2 = to_bytes_and_size(in2)
    assert elemsize == elemsize2

    startbyte1 = start1 * elemsize
    sizebyte1 = size1 * elemsize
    startbyte2 = start2 * elemsize
    sizebyte2 = size2 * elemsize
    assert 0 <= startbyte1 < len(in1)
    assert 0 <= startbyte1 + sizebyte1 <= len(in1)
    assert 0 <= startbyte2 < len(in2)
    assert 0 <= startbyte2 + sizebyte2 <= len(in2)

    buf1 = ffi.from_buffer(in1)
    buf2 = ffi.from_buffer(in2)
    out = ffi.new("char[]", max(sizebyte1, sizebyte2))
    count = lib.intersection(buf1 + startbyte1, sizebyte1, buf2 + startbyte2, sizebyte2, out, elemsize)
    return SlowDiskIntArray(ffi.buffer(out, count), elemsize, sys.byteorder)
