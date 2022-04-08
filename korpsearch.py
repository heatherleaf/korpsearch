
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
        t0 = time.time()
        index = {}
        for n, sentence in enumerate(corpus.sentences(), 1):   # number sentences from 1
            for i in range(len(sentence)):
                key = self._template_key(sentence, i)
                if key:
                    if key not in index:
                        index[key] = set()
                    index[key].add(n)
        self._index.clear()
        for m in index:
            self._index[m] = index[m]
        log(f"{self} --> n:o keys = {len(self._index)}", self._verbose, start=t0)
        self.close()


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
            self._dimensions['index_bytesize'] = self._dimensions['key_bytesize'] + self._dimensions['ptr_bytesize']
            self._index.seek(0, os.SEEK_END)
            self._dimensions['index_size'] = self._index.tell() // self._dimensions['index_bytesize']
            self._sets.seek(0, os.SEEK_END)
            self._dimensions['sets_size'] = self._sets.tell() // self._dimensions['elem_bytesize']

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
        return IndexSet(self._sets, self._dimensions['elem_bytesize'], set_start)

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

    def _count_sentences_and_instances(self, corpus, unsorted_tmpfile, sorted_tmpfile):
        t0 = time.time()
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
        return nr_sentences, nr_instances

    def _build_file_of_key_sentence_pairs(self, corpus, unsorted_tmpfile, sorted_tmpfile):
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
        sets_size = self._count_lines_in_file(sorted_tmpfile)
        log(f" -> sorted {sets_size} unique key-sentence pairs", self._verbose, start=t0)
        return sets_size

    def _write_key_to_indexfile(self, current, key):
        raise NotImplementedError()

    def _build_indexfile_and_setfile(self, sorted_tmpfile):
        t0 = time.time()
        nr_keys = nr_elements = 0
        with open(sorted_tmpfile) as TMP:
            current = set_start = set_size = -1
            # dummy sentence to account for null pointers:
            self._sets.write((0).to_bytes(self._dimensions['elem_bytesize'], byteorder=ENDIANNESS))
            nr_elements += 1
            for line in TMP:
                key, sent = map(int, line.rsplit(maxsplit=1))
                if current < key:
                    if set_start >= 0:
                        assert current >= 0 and set_size > 0
                        # Now the set is full, and we can write the size of the set at its beginning
                        self._sets.seek(set_start)
                        self._sets.write(set_size.to_bytes(self._dimensions['elem_bytesize'], byteorder=ENDIANNESS))
                        self._sets.seek(0, os.SEEK_END)
                    self._write_key_to_indexfile(current, key)
                    self._index.write(nr_elements.to_bytes(self._dimensions['ptr_bytesize'], byteorder=ENDIANNESS))
                    # Add a placeholder for the size of the set
                    set_start, set_size = self._sets.tell(), 0
                    self._sets.write(set_size.to_bytes(self._dimensions['elem_bytesize'], byteorder=ENDIANNESS))
                    nr_elements += 1
                    current = key
                    nr_keys += 1
                self._sets.write(sent.to_bytes(self._dimensions['elem_bytesize'], byteorder=ENDIANNESS))
                set_size += 1
                nr_elements += 1
            # Write the size of the final set at its beginning
            self._sets.seek(set_start)
            self._sets.write(set_size.to_bytes(self._dimensions['elem_bytesize'], byteorder=ENDIANNESS))
            self._sets.seek(0, os.SEEK_END)
        log(f" -> created index file with {nr_keys} keys, sets file with {nr_elements} elements", self._verbose, start=t0)

    def build_index(self, corpus, load_factor=1.0, keep_tmpfiles=False):
        log(f"Building index for {self}", self._verbose)
        unsorted_tmpfile = self._basefile().with_suffix('.unsorted.tmp')
        sorted_tmpfile = self._basefile().with_suffix('.sorted.tmp')
        nr_sentences, nr_instances = self._count_sentences_and_instances(corpus, unsorted_tmpfile, sorted_tmpfile)
        self._dimensions = {}
        self._dimensions['elem_bytesize'] = self._min_bytes_to_store_values(nr_sentences + 1)   # +1 because we number sentences from 1
        self._dimensions['key_bytesize'] = self._min_bytes_to_store_values(nr_instances)
        self._dimensions['max_key_size'] = self._get_max_key_size(nr_instances, load_factor)
        size_of_setsfile = self._build_file_of_key_sentence_pairs(corpus, unsorted_tmpfile, sorted_tmpfile)
        size_of_setsfile += 1 + nr_instances  # +1 because pointer 0 has a special meaning, +nr_instances because every set has a size
        self._dimensions['ptr_bytesize'] = self._min_bytes_to_store_values(size_of_setsfile)
        with open(self._basefile().with_suffix('.dim'), 'w') as DIM:
            json.dump(self._dimensions, DIM)
        self._build_indexfile_and_setfile(sorted_tmpfile)
        log(" -> dimensions: " + ", ".join(f"{k}={v}" for k, v in self._dimensions.items()), self._verbose)
        log("", self._verbose)
        if not keep_tmpfiles:
            unsorted_tmpfile.unlink()
            sorted_tmpfile.unlink()
        self.close()


################################################################################
## Optimised index, using binary search

class BinsearchIndex(SplitIndex):
    dir_suffix = '.indexes.binsearch'

    def _get_max_key_size(self, nr_keys, load_factor):
        return 2 ** (8 * self._dimensions['key_bytesize'])

    def _lookup_key(self, key):
        # store in local variables for faster access
        index_size = self._dimensions['index_size']
        index_bytesize = self._dimensions['index_bytesize']
        key_bytesize = self._dimensions['key_bytesize']
        ptr_bytesize = self._dimensions['ptr_bytesize']
        # binary search
        start, end = 0, index_size - 1
        while start <= end:
            mid = (start + end) // 2
            bytepos = mid * index_bytesize
            self._index.seek(bytepos)
            key0 = int.from_bytes(self._index.read(key_bytesize), byteorder=ENDIANNESS)
            if key0 == key:
                return int.from_bytes(self._index.read(ptr_bytesize), byteorder=ENDIANNESS)
            elif key0 < key:
                start = mid + 1
            else:
                end = mid - 1
        raise ValueError("Key not found")

    def _write_key_to_indexfile(self, current, key):
        self._index.write(key.to_bytes(self._dimensions['key_bytesize'], byteorder=ENDIANNESS))


################################################################################
## Disk-based hash table

class HashIndex(SplitIndex):
    dir_suffix = '.indexes.hashtable'

    def _get_max_key_size(self, nr_keys, load_factor):
        return math.ceil(nr_keys / load_factor)

    def _lookup_key(self, key):
        # store in local variables for faster access
        index_size = self._dimensions['index_size']
        ptr_bytesize = self._dimensions['ptr_bytesize']
        # find the start position in the table
        bytepos = key * ptr_bytesize
        self._index.seek(bytepos)
        return int.from_bytes(self._index.read(ptr_bytesize), byteorder=ENDIANNESS)

    def _write_key_to_indexfile(self, current, key):
        null_ptr = (0).to_bytes(self._dimensions['ptr_bytesize'], byteorder=ENDIANNESS)
        for _ in range(current + 1, key):
            self._index.write(null_ptr)


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
        if self.values:
            return len(self.values)
        return self.size

    def __str__(self):
        MAX = 5
        if len(self) <= MAX:
            return "{" + ", ".join(str(n) for n in self) + "}"
        return f"{{{', '.join(str(n) for n in itertools.islice(self, MAX))}, ... (N={len(self)})}}"

    def __iter__(self):
        if self.values:
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
        elif self.values:
            # O(self + other)
            self.values.intersection_update(set(other))
        else:
            # O(self + other)
            self.values = set(self) & set(other)
        if not self.values:
            raise ValueError("Empty intersection")
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
## Algorithms

ALGORITHMS = {
    'hashtable': HashIndex,
    'binsearch': BinsearchIndex,
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
    log("", args.verbose)
    log(f"Result: {result}", args.verbose, start=starttime)
    print(result)
    for index, _, _ in search_results:
        index.close()


################################################################################
## Main

def build_indexes(args):
    index_class = ALGORITHMS[args.algorithm]
    basedir = args.corpus.with_suffix(index_class.dir_suffix)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    corpus = Corpus(args.corpus, args.features)
    t0 = time.time()
    ctr = 0
    for template in yield_templates(args.features, args.max_dist):
        index_class(args.corpus, template, mode='w', verbose=args.verbose).build_index(corpus)
        ctr += 1
    log(f"Created {ctr} indexes", args.verbose, start=t0)


class Corpus:
    def __init__(self, corpus_file, features):
        self.corpus_file = open(corpus_file)
        self.features = features
        self.reset()

    def reset(self):
        self.corpus_file.seek(0)
        self.corpus_features = {feat : n for n, feat in enumerate(self.corpus_file.readline().split())}

    def sentences(self):
        self.reset()
        sentence = None
        for line in self.corpus_file:
            line = line.strip()
            if line.startswith("# sentence"):
                if sentence:
                    yield sentence
                sentence = []
            elif line:
                values = line.split('\t')
                token = {f : values[self.corpus_features[f]] for f in self.features}
                sentence.append(token)
            elif sentence:
                yield sentence
                sentence = []


def yield_templates(features, max_dist):
    for feat in features:
        templ = [(feat, 0)]
        yield templ
        for feat1 in features:
            for dist in range(1, max_dist+1):
                templ = [(feat, 0), (feat1, dist)]
                yield templ


parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--algorithm', '-a', choices=list(ALGORITHMS), default='binsearch', 
                    help='which lookup algorithm/data structure to use')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
pgroup = parser.add_mutually_exclusive_group(required=True)
pgroup.add_argument('query', nargs='?', help='the query')
pgroup.add_argument('--build-index', action='store_true', help='build the indexes')
parser.add_argument('--features', '-f', nargs='*', help='features')
parser.add_argument('--max-dist', default=2, help='max distance between token pairs')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.build_index:
        if not args.features:
            parser.error("You must specify some --features")
        build_indexes(args)
    else:
        query_corpus(args)
