from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("int intersection(const char *in1, int len1, const char *in2, int len2, char *out, int size);")
ffibuilder.set_source("_merge", '#include "merge.h"', sources=['merge.c'])
ffibuilder.compile(verbose=True)

import _merge

