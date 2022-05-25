
import itertools
from disk import DiskIntArray
from util import bytesify

################################################################################
## Templates and instances

class Template:
    def __init__(self, *feature_positions):
        self._feature_positions = [(bytesify(feat), pos) for feat, pos in feature_positions]

    def __bytes__(self):
        return b'-'.join(feat + str(pos).encode() for feat, pos in self._feature_positions)

    def __str__(self):
        return '-'.join(feat.decode() + str(pos) for feat, pos in self._feature_positions)

    def __iter__(self):
        yield from self._feature_positions

    def __len__(self):
        return len(self._feature_positions)


class Instance:
    def __init__(self, *values):
        self._values = values

    def values(self):
        return self._values

    def __bytes__(self):
        return b' '.join(map(bytesify, self._values))

    def __str__(self):
        return bytes.decode(bytes(self))

    def __iter__(self):
        yield from self._values

    def __len__(self):
        return len(self._values)


################################################################################
## Inverted sentence index
## Implemented as a sorted array of interned strings

class Index:
    dir_suffix = '.indexes'

    def __init__(self, corpus, template, mode='r'):
        assert mode in "rw"
        assert isinstance(template, Template)
        self.basedir = corpus.path().with_suffix(self.dir_suffix)
        self.corpus = corpus
        self.template = template
        basefile = self.basefile()

        self._keypaths = [basefile.with_suffix(f'.{feature.decode()}{pos}') for feature, pos in template]
        self._indexpath = basefile.with_suffix('.index')
        self._setspath = basefile.with_suffix('.sets')

        if mode == 'r':
            self._keys = [DiskIntArray(path) for path in self._keypaths]
            self._index = DiskIntArray(self._indexpath)
            self._sets = DiskIntArray(self._setspath)

    def __str__(self):
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self):
        return len(self._index)

    def basefile(self):
        return self.basedir / str(self.template)

    def close(self):
        for keyarray in self._keys: self._close(keyarray)
        self._close(self._index)
        self._close(self._sets)

        self._keys = []
        self._index = None
        self._sets = None

    def _close(self, file):
        if hasattr(file, "close"):
            file.close()

    def search(self, instance):
        set_start = self._lookup_instance(instance)
        return IndexSet(self._sets, set_start)

    def _lookup_instance(self, instance):
        # binary search
        instance_key = tuple(str.index for str in instance)
        start, end = 0, len(self)-1
        while start <= end:
            mid = (start + end) // 2
            key = tuple(keyarray[mid] for keyarray in self._keys)
            if key == instance_key:
                return self._index[mid]
            elif key < instance_key:
                start = mid + 1
            else:
                end = mid - 1
        raise KeyError(f'Instance "{instance}" not found')


################################################################################
## Index set

class IndexSet:
    def __init__(self, setsarray, start, use_list=False):
        self._setsarray = setsarray
        self._use_list = use_list
        self.start = start
        if start is None:
            self.size = 0
        else:
            self.size = self._setsarray[start]
            self.start += 1
        self.values = None

    def __len__(self):
        if self.values is not None:
            return len(self.values)
        return self.size

    def __str__(self):
        MAX = 5
        if len(self) <= MAX:
            return "{" + ", ".join(str(n) for n in self) + "}"
        return f"{{{', '.join(str(n) for n in itertools.islice(self, MAX))}, ... (N={len(self)})}}"

    def __iter__(self):
        if self.values is not None:
            yield from self.values
        else:
            yield from self._setsarray[self.start:self.start+self.size]

    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection
    _min_size_difference = 1000

    def intersection_update(self, other):
        # We assume that self is smaller than other!
        if len(other) > len(self) * self._min_size_difference:
            # O(self * log(other))
            self.values = [elem for elem in self if elem in other]
        elif isinstance(self.values, set):
            # O(self + other)
            self.values.intersection_update(other)
        else:
            # O(self + other)
            # The result can be a set or a list (sets seem to be 25-50% faster, 
            # but lists are easier to reimplement in C, and to store externally)
            # (It seems like lists are faster with PyPy, but sets with CPython)
            result = [] if self._use_list else set()
            add_result = result.append if self._use_list else result.add
            selfiter, otheriter = iter(sorted(self)), iter(other)
            selfval, otherval = next(selfiter), next(otheriter)
            while True:
                try:
                    if selfval == otherval:
                        add_result(selfval)
                        selfval = next(selfiter)
                        otherval = next(otheriter)
                    elif selfval < otherval:
                        selfval = next(selfiter)
                    else: # selfval > otherval
                        otherval = next(otheriter)
                except StopIteration:
                    break
            self.values = result
        if not self.values:
            raise ValueError("Empty intersection")
        self.start = self.size = self._setsarray = None

    def filter(self, check):
        result_class = list if self._use_list else set
        self.values = result_class(elem for elem in self if check(elem))
        self.start = self.size = self._setsarray = None

    def __contains__(self, elem):
        if isinstance(self.values, set):
            return elem in self.values
        values = self._setsarray if self.values is None else self.values
        start = self.start or 0
        end = start + self.size - 1
        while start <= end:
            mid = (start + end) // 2
            elem0 = values[mid]
            if elem0 == elem:
                return True
            elif elem0 < elem:
                start = mid + 1
            else:
                end = mid - 1
        return False

