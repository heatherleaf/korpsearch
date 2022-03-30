
import os
import sys
import time
import math
import re
import shutil
import argparse
import shelve
import subprocess
import itertools
from pathlib import Path


################################################################################
## Shelf index

class ShelfIndex:
    dir_suffix = '.shelves'

    # class variable to keep track of open shelves
    # (the shelve module doesn't allow several open instances of the same file)
    _open_shelves = {}

    def __init__(self, corpus_name, template, mode='r'):
        self.basedir = corpus_name.with_suffix(self.dir_suffix)
        self.template = template
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
        return self._index_size

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
        index = {}
        for n, sentence in enumerate(corpus):
            for i in range(len(sentence)):
                key = self._template_key(sentence, i)
                if key:
                    if key not in index:
                        index[key] = set()
                    index[key].add(n)
        self._index.clear()
        for m in index:
            self._index[m] = index[m]
        print(f"{self} --> {len(self._index)}")
        self.close()


################################################################################
## Optimised index

# TODO: hash to pointer indexes - then we don't need binary search for looking up a key (and we save space)

ENDIANNESS = 'little'

class BinsearchIndex:
    dir_suffix = '.indexes'

    def __init__(self, corpus_name, template, mode='r'):
        assert mode in "rw"
        self.basedir = corpus_name.with_suffix(self.dir_suffix)
        self.template = template
        basefile = self._basefile()
        self._index = open(basefile.with_suffix('.index'), mode + 'b')
        self._sets = open(basefile.with_suffix('.sets'), mode + 'b')
        if mode == 'r':
            with open(basefile.with_suffix('.dim')) as DIM:
                self._key_bytesize, self._ptr_bytesize, self._elem_bytesize = map(int, DIM.read().split())
            self._index_bytesize = self._key_bytesize + self._ptr_bytesize
            self._key_truncate = (1 << 8 * self._key_bytesize) - 1
            self._index.seek(0, os.SEEK_END)
            self._index_size = self._index.tell() // self._index_bytesize
            self._sets.seek(0, os.SEEK_END)
            self._sets_size = self._sets.tell() // self._elem_bytesize

    def close(self):
        self._index.close()
        self._sets.close()
        self._index = self._sets = None

    def __len__(self):
        return self._index_size

    def __str__(self):
        return "-".join(f"{feat}{k}" for (feat, k) in self.template)

    def _basefile(self):
        return self.basedir / self.__str__()

    def _instance_key(self, instance):
        # Better, use this: https://github.com/flier/pyfasthash
        # (We cannot use Python's hash(), because they use randomness)
        h = 5381  # 5381 is from djb2
        for term in instance:
            for c in term:
                h = h * 65587 + ord(c)  # 65587 is from https://github.com/davidar/sdbm/blob/29d5ed2b5297e51125ee45f6efc5541851aab0fb/hash.c#L16
        return self._key_truncate & h

    def _template_key(self, sentence, k):
        try:
            return self._instance_key(sentence[k+i][feat] for (feat, i) in self.template)
        except IndexError:
            return None

    def _binary_search(self, key):
        start = 0
        end = self._index_size - 1
        while start <= end:
            mid = (start + end) // 2
            key0, set_start = self._lookup(mid)
            if key0 == key:
                if mid+1 < self._index_size:
                    _, set_end = self._lookup(mid+1)
                else:
                    set_end = self._sets_size
                return set_start, set_end
            elif key0 < key:
                start = mid + 1
            else:
                end = mid - 1
        raise ValueError("Key not found")

    def _lookup(self, n):
        bytepos = n * self._index_bytesize
        self._index.seek(bytepos)
        key = int.from_bytes(self._index.read(self._key_bytesize), byteorder=ENDIANNESS)
        ptr = int.from_bytes(self._index.read(self._ptr_bytesize), byteorder=ENDIANNESS)
        return key, ptr

    def search(self, instance):
        key = self._instance_key(instance)
        start, end = self._binary_search(key)
        return IndexSet(self._sets, self._elem_bytesize, start, end)

    def build_index(self, corpus, key_bytesize=2, keep_tmpfiles=False):
        t0 = time.time()
        self._key_truncate = (1 << 8 * key_bytesize) - 1
        elem_bytesize = math.ceil(math.log(len(corpus),2)/8)
        unsorted_tmpfile = self._basefile().with_suffix('.unsorted.tmp')
        sorted_tmpfile = self._basefile().with_suffix('.sorted.tmp')
        ctr_tmp = 0
        with open(unsorted_tmpfile, 'w') as TMP:
            for n, sentence in enumerate(corpus):
                for k in range(len(sentence)):
                    key = self._template_key(sentence, k)
                    if key:
                        print(f"{key}\t{n}", file=TMP)
                        ctr_tmp += 1
        subprocess.run(['sort', '--numeric-sort', '--key=1', '--key=2', '--unique', '--output', sorted_tmpfile, unsorted_tmpfile])
        wc_output = subprocess.run(['wc', '-l', sorted_tmpfile], capture_output=True).stdout
        sets_size = int(wc_output.split()[0])
        ptr_bytesize = math.ceil(math.log(sets_size,2)/8)
        with open(self._basefile().with_suffix('.dim'), 'w') as DIM:
            print(key_bytesize, ptr_bytesize, elem_bytesize, file=DIM)
        ctr_sets = ctr_index = 0
        with open(sorted_tmpfile) as TMP:
            current = None
            for ptr, line in enumerate(TMP):
                key, n = map(int, line.rsplit(maxsplit=1))
                self._sets.write(n.to_bytes(elem_bytesize, byteorder=ENDIANNESS))
                ctr_sets += 1
                if key != current:
                    self._index.write(key.to_bytes(key_bytesize, byteorder=ENDIANNESS))
                    self._index.write(ptr.to_bytes(ptr_bytesize, byteorder=ENDIANNESS))
                    current = key
                    ctr_index += 1
        assert sets_size == ctr_sets, (sets_size, ctr_sets)
        print(f"{self} --> {ctr_tmp} --> {ctr_sets} --> {ctr_index}     # {time.time()-t0:.3f} s")
        if not keep_tmpfiles:
            unsorted_tmpfile.unlink()
            sorted_tmpfile.unlink()
        self.close()


################################################################################
## Index set

class IndexSet:
    def __init__(self, setsfile, elem_bytesize, start, end):
        self._setsfile = setsfile
        self._elem_bytesize = elem_bytesize
        self.start = start
        self.end = end
        self.values = None

    @staticmethod
    def from_set(set):
        iset = IndexSet(None, None, None, None)
        iset.values = set
        return iset

    def __len__(self):
        if self.values:
            return len(self.values)
        return self.end - self.start

    def __str__(self):
        MAX = 5
        if len(self) <= MAX:
            return "{" + ", ".join(str(n) for n in self) + "}"
        return f"{{{', '.join(str(n) for n in itertools.islice(self, MAX))}, ... (N={len(self)})}}"

    def __iter__(self):
        if self.values:
            # return iter(self.values) # <-- doesn't work together with __str__, don't know why
            yield from self.values
        else:
            bytepos = self.start * self._elem_bytesize
            self._setsfile.seek(bytepos)
            for _ in range(self.start, self.end):
                yield int.from_bytes(self._setsfile.read(self._elem_bytesize), byteorder=ENDIANNESS)

    # if the sets have very uneven size, use __contains__ on the larger set
    # instead of normal set intersection
    _min_size_difference = 100

    def intersection_update(self, other):
        if len(other) > len(self) * self._min_size_difference:
            # O(self * log(other))
            self.values = {elem for elem in self if elem in other}
        elif self.values:
            # O(self + other)
            self.values.intersection_update(set(other))
        else:
            # O(self + other)
            self.values = set(self) & set(other)
        if not self.values:
            raise ValueError("Empty intersection")
        self.start = self.end = self._setsfile = None

    def __contains__(self, elem):
        if self.values:
            return elem in self.values
        start = self.start
        end = self.end - 1
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
        bytepos = n * self._elem_bytesize
        self._setsfile.seek(bytepos)
        return int.from_bytes(self._setsfile.read(self._elem_bytesize), byteorder=ENDIANNESS)


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
                    #if i+dist < len(self.query):
                        for feat1, value1 in self.query[i+dist]:
                            templ = [(feat,0), (feat1,dist)]
                            yield (templ, [value, value1])
        # Single tokens: yield subqueries after more complex queries!
        for tok in self.query:
            for feat, value in tok:
                yield ([(feat,0)], [value])

    @staticmethod
    def is_subquery(subtemplate, subinstance, template, instance):
        subquery = [(feat, val) for ((feat, _), val) in zip(subtemplate, subinstance)]
        query = [(feat, val) for ((feat, _), val) in zip(template, instance)]
        if len(subquery) > 1: return False
        return subquery[0] in query


def query_corpus(args):
    index_class = ShelfIndex if args.use_shelf else BinsearchIndex
    query = Query(args.query)
    print("Query:", args.query, "-->", query)
    starttime = time.time()
    print("Searching:")
    t0 = time.time()
    search_results = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = index_class(args.corpus, template)
        except FileNotFoundError:
            continue
        sentences = index.search(instance)
        print(f"   {index} = {'-'.join(instance)} --> {len(sentences)}")
        search_results.append((index, instance, sentences))
    print(f"   # {time.time()-t0:.3f} s")
    print("Sorting:")
    t0 = time.time()
    search_results.sort(key=lambda r: len(r[-1]))
    # search_results.sort(key=lambda r: -len(r[-1]))
    print("   ", " ".join(str(index) for index, _, _ in search_results), f"  # {time.time()-t0:.3f} s")
    print("Intersecting:")
    result = None
    for index, instance, sentences in search_results:
        t0 = time.time()
        if result is None:
            result = sentences
        else:
            result.intersection_update(sentences)
        print(f"   {index} = {'-'.join(instance)} : {len(sentences)} --> {result}   # {time.time()-t0:.3f} s")
    print()
    print(f"Time: {time.time()-starttime:.3f} s")
    print("Result:", result)
    for index, _, _ in search_results:
        index.close()


################################################################################
## Main

def build_indexes(args):
    index_class = ShelfIndex if args.use_shelf else BinsearchIndex
    basedir = args.corpus.with_suffix(index_class.dir_suffix)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    t0 = time.time()
    corpus = read_corpus(args.corpus, args.features)
    print(f"Corpus: {len(corpus)} sentences, {sum(map(len, corpus))} tokens   # {time.time()-t0:.3f} s")
    for template in yield_templates(args.features, args.max_dist):
        index_class(args.corpus, template, mode='w').build_index(corpus)


def read_corpus(corpus_file, features):
    sentences = []
    with open(corpus_file) as IN:
        corpus_features = {feat : n for n, feat in enumerate(IN.readline().split())}
        for k, line in enumerate(IN):
            line = line.strip()
            if line.startswith("# sentence"):
                sentences.append([])
            elif line and sentences:
                values = line.split('\t')
                token = {f : values[corpus_features[f]] for f in features}
                sentences[-1].append(token)
    return sentences


def yield_templates(features, max_dist):
    for feat in features:
        templ = [(feat, 0)]
        yield templ
        for feat1 in features:
            for dist in range(1, max_dist+1):
                templ = [(feat, 0), (feat1, dist)]
                yield templ


parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--use-shelf', '-s', action='store_true', help='use the `shelve` library')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
pgroup = parser.add_mutually_exclusive_group(required=True)
pgroup.add_argument('query', nargs='?', help='the query')
pgroup.add_argument('--build-index', action='store_true', help='build the indexes')
parser.add_argument('--features', '-f', nargs='*', help='features')
parser.add_argument('--max-dist', default=2, help='max distance between token pairs')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.build_index:
        if not args.features:
            parser.error("You must specify some --features")
        build_indexes(args)
    else:
        query_corpus(args)
