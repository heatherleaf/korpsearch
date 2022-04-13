
import os
import time
import math
import re
import json
import shutil
import argparse
import shelve
import subprocess
import itertools
from pathlib import Path


def log(output, verbose, start=None):
    if verbose:
        if start:
            duration = time.time()-start
            print(output.ljust(100), f"{duration//60:4.0f}:{duration%60:05.2f}")
        else:
            print(output)


################################################################################
## Shelf index

class ShelfIndex:
    dir_suffix = '.indexes.shelve'

    # class variable to keep track of open shelves
    # (the shelve module doesn't allow several open instances of the same file)
    _open_shelves = {}

    def __init__(self, corpus_name, template, mode='r', verbose=False):
        self.basedir = corpus_name.with_suffix(self.dir_suffix)
        self.template = template
        self._verbose = verbose
        ifile = str(self._indexfile())
        if mode == 'r' and not self._indexfile().is_file():
            raise FileNotFoundError(f"No such database file: '{ifile}'")
        if ifile in ShelfIndex._open_shelves:
            self._index = ShelfIndex._open_shelves[ifile]
        else:
            self._index = shelve.open(ifile)
            if mode == 'r' and len(self._index) == 0:
                raise FileNotFoundError(f"No such database file: '{ifile}'")
            ShelfIndex._open_shelves[ifile] = self._index

    def close(self):
        ifile = self._indexfile()
        if ifile in ShelfIndex._open_shelves:
            del ShelfIndex._open_shelves[ifile]
            self._index.close()

    def __len__(self):
        return len(self._index)

    def __str__(self):
        return "-".join(f"{feat}{k}" for (feat, k) in self.template)

    def _indexfile(self):
        return self.basedir / self.__str__()

    def _instance_key(self, instance):
        return ' '.join(instance)

    def _template_key(self, sentence, k):
        try:
            return self._instance_key(sentence[k+i][feat] for (feat, i) in self.template)
        except IndexError:
            return None

    def search(self, instance):
        key = self._instance_key(instance)
        result = self._index[key]
        return IndexSet.from_set(result)

    def build_index(self, corpus):
        log(f"Building index for {self}", self._verbose)
        t0 = time.time()
        index = {}
        for n, sentence in enumerate(corpus.sentences(), 1):   # number sentences from 1
            for i in range(len(sentence)):
                key = self._template_key(sentence, i)
                if key:
                    if key not in index:
                        index[key] = set()
                    index[key].add(n)
        log(f" -> created {len(index)} keys in internal dict", self._verbose, start=t0)
        t0 = time.time()
        self._index.clear()
        for m in index:
            self._index[m] = index[m]
        self.close()
        size = self._indexfile().stat().st_size / 1024 / 1024
        log(f" -> created shelve db ({size:.1f} mb)", self._verbose, start=t0)
        log("", self._verbose)


################################################################################
## Splitting the index in two files
## Note: This is an abstract class!

ENDIANNESS = 'little'

class SplitIndex:
    def __init__(self, corpus_name, template, mode='r', verbose=False):
        assert mode in "rw"
        self.basedir = corpus_name.with_suffix(self.dir_suffix)
        self.template = template
        self._verbose = verbose
        basefile = self._basefile()
        self._index = open(basefile.with_suffix('.index'), mode + 'b')
        self._sets = open(basefile.with_suffix('.sets'), mode + 'b')
        if mode == 'r':
            with open(basefile.with_suffix('.dim')) as DIM:
                self._dimensions = json.load(DIM)
            self._dimensions['index_bytes'] = self._dimensions['key_bytes'] + self._dimensions['ptr_bytes']
            self._index.seek(0, os.SEEK_END)
            self._dimensions['index_size'] = self._index.tell() // self._dimensions['index_bytes']
            self._sets.seek(0, os.SEEK_END)
            self._dimensions['sets_size'] = self._sets.tell() // self._dimensions['elem_bytes']

    def close(self):
        self._index.close()
        self._sets.close()
        self._index = self._sets = None

    def __len__(self):
        return self._dimensions['index_size']

    def __str__(self):
        return "-".join(f"{feat}{k}" for (feat, k) in self.template)

    def _basefile(self):
        return self.basedir / self.__str__()

    def _instance_key(self, instance):
        h = 5381  # 5381 is from djb2:https://stackoverflow.com/questions/10696223/reason-for-the-number-5381-in-the-djb-hash-function
        for term in instance:
            for c in term:
                h = h * 65587 + ord(c)  # 65587 is from https://github.com/davidar/sdbm/blob/29d5ed2b5297e51125ee45f6efc5541851aab0fb/hash.c#L16
        return h % self._dimensions['max_key_size']

    def _yield_instances(self, sentence):
        for k in range(len(sentence)):
            try:
                yield [sentence[k+i][feat] for (feat, i) in self.template]
            except IndexError:
                pass

    def search(self, instance):
        key = self._instance_key(instance)
        set_start = self._lookup_key(key)
        return IndexSet(self._sets, self._dimensions['elem_bytes'], set_start)

    def _lookup_key(self, key):
        raise NotImplementedError()

    def _get_max_key_size(self, nr_keys, load_factor):
        raise NotImplementedError()

    def _min_bytes_to_store_values(self, nr_values):
        return math.ceil(math.log(nr_values, 2) / 8)

    def _sort_file_unique(self, unsorted_tmpfile, sorted_tmpfile, *extra_args):
        subprocess.run(['sort'] + list(extra_args) + ['--unique', '--output', sorted_tmpfile, unsorted_tmpfile])

    def _count_lines_in_file(self, file):
        wc_output = subprocess.run(['wc', '-l', file], capture_output=True).stdout
        return int(wc_output.split()[0])

    def _write_key_to_indexfile(self, current, key):
        raise NotImplementedError()

    def build_index(self, corpus, load_factor=1.0, keep_tmpfiles=False):
        log(f"Building index for {self}", self._verbose)
        unsorted_tmpfile = self._basefile().with_suffix('.unsorted.tmp')
        sorted_tmpfile = self._basefile().with_suffix('.sorted.tmp')

        # Count sentences and instances
        start_time = t0 = time.time()
        nr_sentences = 0
        nr_lines = 0
        with open(unsorted_tmpfile, 'w') as TMP:
            for sentence in corpus.sentences():
                nr_sentences += 1
                for instance in self._yield_instances(sentence):
                    print(" ".join(instance), file=TMP)
                    nr_lines += 1
        log(f" -> created unsorted instances file, {nr_lines} lines, {nr_sentences} sentences", self._verbose, start=t0)
        t0 = time.time()
        subprocess.run(['sort', '--unique', '--output', sorted_tmpfile, unsorted_tmpfile])
        self._sort_file_unique(unsorted_tmpfile, sorted_tmpfile)
        nr_instances = self._count_lines_in_file(sorted_tmpfile)
        log(f" -> sorted {nr_instances} unique instances", self._verbose, start=t0)

        # Calculate dimensions
        self._dimensions = {}
        self._dimensions['elem_bytes'] = self._min_bytes_to_store_values(nr_sentences + 1)   # +1 because we number sentences from 1
        self._dimensions['key_bytes'] = self._min_bytes_to_store_values(nr_instances)
        self._dimensions['max_key_size'] = self._get_max_key_size(nr_instances, load_factor)

        # Build file of key-sentence pairs
        t0 = time.time()
        nr_lines = 0
        with open(unsorted_tmpfile, 'w') as TMP:
            for n, sentence in enumerate(corpus.sentences(), 1):   # number sentences from 1
                for instance in self._yield_instances(sentence):
                    key = self._instance_key(instance)
                    print(f"{key}\t{n}", file=TMP)
                    nr_lines += 1
        log(f" -> created unsorted key-sentence file, {nr_lines} lines", self._verbose, start=t0)
        t0 = time.time()
        self._sort_file_unique(unsorted_tmpfile, sorted_tmpfile, '--numeric-sort', '--key=1', '--key=2')
        size_of_setsfile = self._count_lines_in_file(sorted_tmpfile)
        log(f" -> sorted {size_of_setsfile} unique key-sentence pairs", self._verbose, start=t0)
        size_of_setsfile += 1 + nr_instances  # +1 because pointer 0 has a special meaning, +nr_instances because every set has a size

        # More dimensions to calculate
        self._dimensions['ptr_bytes'] = self._min_bytes_to_store_values(size_of_setsfile)
        with open(self._basefile().with_suffix('.dim'), 'w') as DIM:
            json.dump(self._dimensions, DIM)
        log(" -> dimensions: " + ", ".join(f"{k}={v}" for k, v in self._dimensions.items()), self._verbose)

        # Build index file and sets file
        t0 = time.time()
        nr_keys = nr_elements = 0
        with open(sorted_tmpfile) as TMP:
            current = set_start = set_size = -1
            # Dummy sentence to account for null pointers:
            self._sets.write((0).to_bytes(self._dimensions['elem_bytes'], byteorder=ENDIANNESS))
            nr_elements += 1
            for line in TMP:
                key, sent = map(int, line.rsplit(maxsplit=1))
                if current < key:
                    if set_start >= 0:
                        assert current >= 0 and set_size > 0
                        # Now the set is full, and we can write the size of the set at its beginning
                        self._sets.seek(set_start)
                        self._sets.write(set_size.to_bytes(self._dimensions['elem_bytes'], byteorder=ENDIANNESS))
                        self._sets.seek(0, os.SEEK_END)
                    self._write_key_to_indexfile(current, key)
                    self._index.write(nr_elements.to_bytes(self._dimensions['ptr_bytes'], byteorder=ENDIANNESS))
                    # Add a placeholder for the size of the set
                    set_start, set_size = self._sets.tell(), 0
                    self._sets.write(set_size.to_bytes(self._dimensions['elem_bytes'], byteorder=ENDIANNESS))
                    nr_elements += 1
                    current = key
                    nr_keys += 1
                self._sets.write(sent.to_bytes(self._dimensions['elem_bytes'], byteorder=ENDIANNESS))
                set_size += 1
                nr_elements += 1
            # Write the size of the final set at its beginning
            self._sets.seek(set_start)
            self._sets.write(set_size.to_bytes(self._dimensions['elem_bytes'], byteorder=ENDIANNESS))
            self._sets.seek(0, os.SEEK_END)
        log(f" -> created index file with {nr_keys} keys, sets file with {nr_elements} elements", self._verbose, start=t0)
        sizes = [f.tell()/1024/1024 for f in (self._index, self._sets)]
        log(f" -> created .index ({sizes[0]:.1f} mb), .sets ({sizes[1]:.1f} mb)", self._verbose, start=start_time)

        # Cleanup
        if not keep_tmpfiles:
            unsorted_tmpfile.unlink()
            sorted_tmpfile.unlink()
        self.close()
        log("", self._verbose)


################################################################################
## Optimised index, using binary search

class BinsearchIndex(SplitIndex):
    dir_suffix = '.indexes.binsearch'

    def _get_max_key_size(self, nr_keys, load_factor):
        return 2 ** (8 * self._dimensions['key_bytes'])

    def _lookup_key(self, key):
        # store in local variables for faster access
        index_size = self._dimensions['index_size']
        index_bytes = self._dimensions['index_bytes']
        key_bytes = self._dimensions['key_bytes']
        ptr_bytes = self._dimensions['ptr_bytes']
        # binary search
        start, end = 0, index_size - 1
        while start <= end:
            mid = (start + end) // 2
            bytepos = mid * index_bytes
            self._index.seek(bytepos)
            key0 = int.from_bytes(self._index.read(key_bytes), byteorder=ENDIANNESS)
            if key0 == key:
                return int.from_bytes(self._index.read(ptr_bytes), byteorder=ENDIANNESS)
            elif key0 < key:
                start = mid + 1
            else:
                end = mid - 1
        raise ValueError("Key not found")

    def _write_key_to_indexfile(self, current, key):
        self._index.write(key.to_bytes(self._dimensions['key_bytes'], byteorder=ENDIANNESS))


################################################################################
## Disk-based hash table

class HashIndex(SplitIndex):
    dir_suffix = '.indexes.hashtable'

    def _get_max_key_size(self, nr_keys, load_factor):
        return math.ceil(nr_keys / load_factor)

    def _lookup_key(self, key):
        # store in local variables for faster access
        index_size = self._dimensions['index_size']
        ptr_bytes = self._dimensions['ptr_bytes']
        # find the start position in the table
        bytepos = key * ptr_bytes
        self._index.seek(bytepos)
        return int.from_bytes(self._index.read(ptr_bytes), byteorder=ENDIANNESS)

    def _write_key_to_indexfile(self, current, key):
        null_ptr = (0).to_bytes(self._dimensions['ptr_bytes'], byteorder=ENDIANNESS)
        for _ in range(current + 1, key):
            self._index.write(null_ptr)


################################################################################
## Binary search index, using instances as keys

class InstanceIndex:
    dir_suffix = '.indexes.instances'

    def __init__(self, corpus, template, mode='r', verbose=False):
        assert mode in "rw"
        self.corpus = Path(corpus)
        self.template = template
        self._verbose = verbose
        basefile = self._basefile()
        self._keys = open(basefile.with_suffix('.keys'), mode + 'b')
        self._index = open(basefile.with_suffix('.index'), mode + 'b')
        self._sets = open(basefile.with_suffix('.sets'), mode + 'b')
        if mode == 'r':
            with open(basefile.with_suffix('.dim')) as DIM:
                self._dimensions = json.load(DIM)

    def close(self):
        self._keys.close()
        self._index.close()
        self._sets.close()
        self._keys = self._index = self._sets = None

    def __len__(self):
        self._keys.seek(0, os.SEEK_END)
        return self._keys.tell() // self._dimensions['keyptr']

    def __str__(self):
        return "-".join(f"{feat}{k}" for (feat, k) in self.template)

    def _basefile(self):
        return self.corpus.with_suffix(self.dir_suffix) / self.__str__()

    def _instance_str(self, instance):
        return instance if isinstance(instance, str) else ' '.join(instance)

    def _yield_instances(self, sentence):
        for k in range(len(sentence)):
            try:
                yield self._instance_str(sentence[k+i][feat] for (feat, i) in self.template)
            except IndexError:
                pass

    def search(self, instance):
        set_start = self._lookup_instance(instance)
        return IndexSet(self._sets, self._dimensions['elemptr'], set_start)

    def _lookup_instance(self, instance):
        instance = self._instance_str(instance)
        # store in local variables for faster access
        keyptr_size = self._dimensions['keyptr']
        setptr_size = self._dimensions['setptr']
        # binary search
        start, end = 0, len(self)-1
        while start <= end:
            mid = (start + end) // 2
            self._keys.seek(mid * keyptr_size)
            keyptr = int.from_bytes(self._keys.read(keyptr_size), byteorder=ENDIANNESS)
            self._index.seek(keyptr)
            keylen = int.from_bytes(self._index.read(1), byteorder=ENDIANNESS)
            instance0 = self._index.read(keylen).decode('utf-8')
            if instance0 == instance:
                set_start = int.from_bytes(self._index.read(setptr_size), byteorder=ENDIANNESS)
                return set_start
            elif instance0 < instance:
                start = mid + 1
            else:
                end = mid - 1
        raise ValueError("Key not found")

    def _min_bytes_to_store_values(self, nr_values):
        return math.ceil(math.log(nr_values, 2) / 8)

    def _sort_file_unique(self, unsorted_file, sorted_file, *extra_args):
        subprocess.run(['sort'] + list(extra_args) + ['--unique', '--output', sorted_file, unsorted_file])

    def _count_lines_in_file(self, file):
        wc_output = subprocess.run(['wc', '-l', file], capture_output=True).stdout
        return int(wc_output.split()[0])

    def build_index(self, corpus, load_factor=1.0, keep_tmpfiles=False):
        log(f"Building index for {self}", self._verbose)
        unsorted_tmpfile = self._basefile().with_suffix('.unsorted.tmp')
        sorted_tmpfile = self._basefile().with_suffix('.sorted.tmp')

        # Count sentences and instances
        start_time = t0 = time.time()
        nr_sentences = 0
        nr_lines = 0
        with open(unsorted_tmpfile, 'w') as TMP:
            for sentence in corpus.sentences():
                nr_sentences += 1
                for instance in self._yield_instances(sentence):
                    print(instance, file=TMP)
                    nr_lines += 1
        log(f" -> created unsorted instances file, {nr_lines} lines, {nr_sentences} sentences", self._verbose, start=t0)
        t0 = time.time()
        subprocess.run(['sort', '--unique', '--output', sorted_tmpfile, unsorted_tmpfile])
        self._sort_file_unique(unsorted_tmpfile, sorted_tmpfile)
        nr_instances = self._count_lines_in_file(sorted_tmpfile)
        size_of_all_instances = sorted_tmpfile.stat().st_size
        log(f" -> sorted {nr_instances} unique instances", self._verbose, start=t0)

        # Build file of key-sentence pairs
        t0 = time.time()
        nr_lines = 0
        with open(unsorted_tmpfile, 'w') as TMP:
            width = math.ceil(math.log(nr_sentences + 1, 10)) + 1
            for n, sentence in enumerate(corpus.sentences(), 1):   # number sentences from 1
                for instance in self._yield_instances(sentence):
                    print(f"{instance}\t{n:0{width}d}", file=TMP)
                    nr_lines += 1
        log(f" -> created unsorted key-sentence file, {nr_lines} lines", self._verbose, start=t0)
        t0 = time.time()
        self._sort_file_unique(unsorted_tmpfile, sorted_tmpfile)
        nr_inst_sent_pairs = self._count_lines_in_file(sorted_tmpfile)
        log(f" -> sorted {nr_inst_sent_pairs} unique instance-sentence pairs", self._verbose, start=t0)

        # Calculate dimensions
        self._dimensions = {}
        self._dimensions['elemptr'] = self._min_bytes_to_store_values(nr_sentences + 1)   # +1 because we number sentences from 1
        self._dimensions['setptr'] = self._min_bytes_to_store_values(nr_inst_sent_pairs)
        indexfile_size = size_of_all_instances + nr_instances * (1 + self._dimensions['setptr'])
        self._dimensions['keyptr'] = self._min_bytes_to_store_values(indexfile_size)
        with open(self._basefile().with_suffix('.dim'), 'w') as DIM:
            json.dump(self._dimensions, DIM)
        log(" -> dimensions: " + ", ".join(f"{k}={v}" for k, v in self._dimensions.items()), self._verbose)

        # Build keys file, index file and sets file
        t0 = time.time()
        nr_keys = nr_elements = 0
        with open(sorted_tmpfile) as IN:
            current = set_start = set_size = None
            # Dummy sentence to account for null pointers:
            self._sets.write((0).to_bytes(self._dimensions['elemptr'], byteorder=ENDIANNESS))
            nr_elements += 1
            for line in IN:
                instance, sent = line.split('\t')
                sent = int(sent)
                if current != instance:
                    if set_start is not None:
                        assert set_size
                        # Now the set is full, and we can write the size of the set at its beginning
                        self._sets.seek(set_start)
                        self._sets.write(set_size.to_bytes(self._dimensions['elemptr'], byteorder=ENDIANNESS))
                        self._sets.seek(0, os.SEEK_END)
                    key = instance.encode('utf-8')
                    assert len(key) <= 255, f"Too long instance: {instance}"
                    keyptr = self._index.tell()
                    self._keys.write(keyptr.to_bytes(self._dimensions['keyptr'], byteorder=ENDIANNESS))
                    self._index.write(len(key).to_bytes(1, byteorder=ENDIANNESS))
                    self._index.write(key)
                    self._index.write(nr_elements.to_bytes(self._dimensions['setptr'], byteorder=ENDIANNESS))
                    # Add a placeholder for the size of the set
                    set_start, set_size = self._sets.tell(), 0
                    self._sets.write(set_size.to_bytes(self._dimensions['elemptr'], byteorder=ENDIANNESS))
                    nr_elements += 1
                    current = instance
                    nr_keys += 1
                self._sets.write(sent.to_bytes(self._dimensions['elemptr'], byteorder=ENDIANNESS))
                set_size += 1
                nr_elements += 1
            # Write the size of the final set at its beginning
            self._sets.seek(set_start)
            self._sets.write(set_size.to_bytes(self._dimensions['elemptr'], byteorder=ENDIANNESS))
            self._sets.seek(0, os.SEEK_END)
        log(f" -> created index file with {nr_keys} keys, sets file with {nr_elements} elements", self._verbose, start=t0)
        sizes = [f.tell()/1024/1024 for f in (self._keys, self._index, self._sets)]
        log(f" -> created .keys ({sizes[0]:.1f} mb), .index ({sizes[1]:.1f} mb), .sets ({sizes[2]:.1f} mb)", self._verbose, start=start_time)

        if not keep_tmpfiles:
            unsorted_tmpfile.unlink()
            sorted_tmpfile.unlink()
        self.close()
        log("", self._verbose)


################################################################################
## Index set

class IndexSet:
    def __init__(self, setsfile, elemsize, start):
        self._setsfile = setsfile
        self._elemsize = elemsize
        self.start = start
        if start is None:
            self.size = 0
        else:
            self._setsfile.seek(start * elemsize)
            self.size = int.from_bytes(self._setsfile.read(self._elemsize), byteorder=ENDIANNESS)
            self.start += 1
        self.values = None

    @staticmethod
    def from_set(set):
        iset = IndexSet(None, None, None)
        iset.values = set
        return iset

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
            self._setsfile.seek(self.start * self._elemsize)
            for _ in range(self.size):
                yield int.from_bytes(self._setsfile.read(self._elemsize), byteorder=ENDIANNESS)

    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection
    _min_size_difference = 100

    def intersection_update(self, other):
        if len(other) > len(self) * self._min_size_difference:
            # O(self * log(other))
            self.values = {elem for elem in self if elem in other}
        else:
            # O(self + other)
            try:
                self.values.intersection_update(set(other))
            except AttributeError:
                self.values = set(self) & set(other)
        if not self.values:
            raise ValueError("Empty intersection")
        self.start = self.size = self._setsfile = None

    def filter(self, check):
        self.values = {elem for elem in self if check(elem)}
        self.start = self.size = self._setsfile = None

    def __contains__(self, elem):
        if self.values:
            return elem in self.values
        start = self.start
        end = start + self.size - 1
        while start <= end:
            mid = (start + end) // 2
            elem0 = self._lookup(mid)
            if elem0 == elem:
                return True
            elif elem0 < elem:
                start = mid + 1
            else:
                end = mid - 1
        return False

    def _lookup(self, n):
        self._setsfile.seek(n * self._elemsize)
        return int.from_bytes(self._setsfile.read(self._elemsize), byteorder=ENDIANNESS)


################################################################################
## Corpus

class Corpus:
    def __init__(self, corpus, mode='r', verbose=False):
        basefile = Path(corpus)
        self._corpus = open(basefile.with_suffix('.csv'))
        self._index = open(basefile.with_suffix('.csv.index'), mode + 'b')
        self._verbose = verbose
        self.reset()

    def close(self):
        self._corpus.close()
        self._index.close()
        self._corpus = self._index = None

    def __str__(self):
        return f"[Corpus: {self._corpus.stem}]"

    def reset(self):
        self._corpus.seek(0)
        # the first line in the CSV should be a header with the names of each column (=features)
        self._features = self._corpus.readline().split()

    def sentences(self):
        self.reset()
        while True:
            _pos, sentence = self._get_next_sentence()
            if sentence is None:
                return
            yield sentence

    def _get_next_sentence(self):
        features = self._features
        corpus = self._corpus
        line = None
        while not line or line.startswith("# sentence"):
            startpos = corpus.tell()
            line = corpus.readline()
            if not line:
                return None, None
            line = line.strip()
        sentence = []
        while line and not line.startswith("# sentence"):
            token = dict(zip(features, line.split('\t')))
            sentence.append(token)
            line = corpus.readline().strip()
        return startpos, sentence

    _pointer_size = 4  # bytes per integer used in the index

    def lookup_sentence(self, n):
        self._index.seek(n * self._pointer_size)
        csv_pos = int.from_bytes(self._index.read(self._pointer_size), byteorder=ENDIANNESS)
        self._corpus.seek(csv_pos)
        _pos, sentence = self._get_next_sentence()
        return sentence

    def build_index(self):
        t0 = time.time()
        self.reset()
        self._index.write((0).to_bytes(self._pointer_size, byteorder=ENDIANNESS))  # number sentences from 1
        ctr = 0
        while True:
            pos, sentence = self._get_next_sentence()
            if sentence is None:
                break
            self._index.write(pos.to_bytes(self._pointer_size, byteorder=ENDIANNESS))
            ctr += 1
        log(f"Built corpus index, {ctr} sentences", self._verbose, start=t0)
        log("", self._verbose)


################################################################################
## Algorithms

ALGORITHMS = {
    'hashtable': HashIndex,
    'binsearch': BinsearchIndex,
    'instance': InstanceIndex,
    'shelve': ShelfIndex,
}

################################################################################
## Queries

QUEREGEX = re.compile(rf'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    def __init__(self, querystr):
        querystr = querystr.replace(' ', '')
        if not QUEREGEX.match(querystr):
            raise ValueError(f"Error in query: {querystr}")
        tokens = querystr.split('][')
        self.query = []
        for tok in tokens:
            self.query.append([])
            parts = re.findall(r'\w+="[^"]+"', tok)
            for part in parts:
                feat, value = part.split('=', 1)
                value = value.replace('"', '')
                self.query[-1].append((feat, value))

    def __str__(self):
        return " ".join("[" + " ".join(f'{feat}="{val}"' for feat, val in subq) + "]" for subq in self.query)

    def subqueries(self):
        # Pairs of tokens
        for i, tok in enumerate(self.query):
            for feat, value in tok:
                for dist in range(1, len(self.query)-i):
                    for feat1, value1 in self.query[i+dist]:
                        templ = [(feat,0), (feat1,dist)]
                        yield (templ, [value, value1])
        # Single tokens: yield subqueries after more complex queries!
        for tok in self.query:
            for feat, value in tok:
                yield ([(feat,0)], [value])

    def features(self):
        return {feat for tok in self.query for feat, _val in tok}

    def check_sentence(self, sentence):
        for k in range(len(sentence) - len(self.query) + 1):
            if all(sentence[k+i][feat] == value 
                   for i, token_query in enumerate(self.query)
                   for feat, value in token_query
                   ):
                return True
        return False

    @staticmethod
    def is_subquery(subtemplate, subinstance, template, instance):
        subquery = [(feat, val) for ((feat, _), val) in zip(subtemplate, subinstance)]
        query = [(feat, val) for ((feat, _), val) in zip(template, instance)]
        if len(subquery) > 1: return False
        return subquery[0] in query


def query_corpus(args):
    index_class = ALGORITHMS[args.algorithm]
    query = Query(args.query)
    starttime = time.time()
    log(f"Query: {args.query} --> {query}", args.verbose)
   
    log("Searching:", args.verbose)
    search_results = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = index_class(args.corpus, template)
        except FileNotFoundError:
            continue
        t0 = time.time()
        sentences = index.search(instance)
        log(f"   {index} = {'-'.join(instance)} --> {len(sentences)}", args.verbose, start=t0)
        search_results.append((index, instance, sentences))

    log("Sorting:", args.verbose)
    t0 = time.time()
    search_results.sort(key=lambda r: len(r[-1]))
    # search_results.sort(key=lambda r: -len(r[-1]))
    log("   " + " ".join(str(index) for index, _, _ in search_results), args.verbose, start=t0)

    log("Intersecting:", args.verbose)
    result = None
    for index, instance, sentences in search_results:
        t0 = time.time()
        if result is None:
            result = sentences
        else:
            result.intersection_update(sentences)
        log(f"   {index} = {'-'.join(instance)} : {len(sentences)} --> {result}", args.verbose, start=t0)

    log("Final filter:", args.verbose)
    t0 = time.time()
    corpus = Corpus(args.corpus)
    result.filter(lambda sent: query.check_sentence(corpus.lookup_sentence(sent)))
    log(f"   {query} --> {result}", args.verbose, start=t0)

    log("", args.verbose)
    log(f"Result: {result}", args.verbose, start=starttime)
    print(result)
    for index, _, _ in search_results:
        index.close()


################################################################################
## Building indexes

def build_indexes(args):
    index_class = ALGORITHMS[args.algorithm]
    basedir = args.corpus.with_suffix(index_class.dir_suffix)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    corpus = Corpus(args.corpus, mode='w', verbose=args.verbose)
    t0 = time.time()
    corpus.build_index()
    ctr = 1
    for template in yield_templates(args.features, args.max_dist):
        index_class(args.corpus, template, mode='w', verbose=args.verbose).build_index(corpus)
        ctr += 1
    log(f"Created {ctr} indexes", args.verbose, start=t0)


def yield_templates(features, max_dist):
    for feat in features:
        templ = [(feat, 0)]
        yield templ
        for feat1 in features:
            for dist in range(1, max_dist+1):
                templ = [(feat, 0), (feat1, dist)]
                yield templ


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--algorithm', '-a', choices=list(ALGORITHMS), default='binsearch', 
                    help='which lookup algorithm/data structure to use')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
pgroup = parser.add_mutually_exclusive_group(required=True)
pgroup.add_argument('query', nargs='?', help='the query')
pgroup.add_argument('--build-index', action='store_true', help='build the indexes')
parser.add_argument('--features', '-f', nargs='*', help='features')
parser.add_argument('--max-dist', type=int, default=2, 
                    help='max distance between token pairs (default: 2)')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.build_index:
        if not args.features:
            parser.error("You must specify some --features")
        build_indexes(args)
    else:
        query_corpus(args)
