
import os
import time
import math
import re
import json
import shutil
import argparse
import sys
import subprocess
import itertools
from pathlib import Path
import mmap


def log(output, verbose, start=None):
    if verbose:
        if start:
            duration = time.time()-start
            print(output.ljust(100), f"{duration//60:4.0f}:{duration%60:05.2f}")
        else:
            print(output)

BYTEORDER = sys.byteorder
BYTES = 4
CASTFORMAT = 'I'

class DiskIntArrayBuilder:
    def __init__(self, filename):
        self._file = open(filename, 'wb')

    def close(self):
        self._file.close()
        self._file = None

    def __len__(self):
        bytepos = self._file.tell()
        assert bytepos % BYTES == 0, (bytepos, bytepos / BYTES)
        return bytepos // BYTES

    def append(self, value):
        self._file.write(value.to_bytes(BYTES, byteorder=BYTEORDER))

    def __setitem__(self, k, value):
        self._file.seek(k * BYTES)
        self._file.write(value.to_bytes(BYTES, byteorder=BYTEORDER))
        self._file.seek(0, os.SEEK_END)


def DiskIntArray(filename):
    file = open(filename, 'rb')
    filearray = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
    return memoryview(filearray).cast(CASTFORMAT)


################################################################################
## Templates and instances

def _bytesify(s):
    assert isinstance(s, (bytes, str)), f"Not str or bytes: {s}"
    return s.encode() if isinstance(s, str) else s


class Template:
    def __init__(self, *feature_positions):
        self._feature_positions = [(_bytesify(feat), pos) for feat, pos in feature_positions]

    def __bytes__(self):
        return b'-'.join(feat + str(pos).encode() for feat, pos in self._feature_positions)

    def __str__(self):
        return '-'.join(feat.decode() + str(pos) for feat, pos in self._feature_positions)

    def __iter__(self):
        yield from self._feature_positions


class Instance:
    def __init__(self, *values):
        self._values = [_bytesify(val) for val in values]

    def __bytes__(self):
        return b' '.join(self._values)

    def __str__(self):
        return ' '.join(map(bytes.decode, self._values))

    def __iter__(self):
        yield from self._values


################################################################################
## Inverted sentence index
## Implemented as an open addressing hash table


class Index:
    dir_suffix = '.hashindex'

    def __init__(self, corpus_name, template, mode='r', verbose=False):
        assert mode in "rw"
        # array_class = DiskIntArray if mode == 'r' else DiskIntArrayBuilder
        array_class = DiskIntArray if mode == 'r' else DiskIntArrayBuilder
        assert isinstance(template, Template)
        self.basedir = corpus_name.with_suffix(self.dir_suffix)
        self.template = template
        self._verbose = verbose
        basefile = self._basefile()
        self._index = array_class(basefile.with_suffix('.index'))
        self._sets = array_class(basefile.with_suffix('.sets'))
        self._index_size = len(self._index)

    def __str__(self):
        return self.__class__.__name__ + ':' + str(self.template) 

    def _basefile(self):
        return self.basedir / str(self.template)

    def _yield_instances(self, sentence):
        try:
            for k in range(len(sentence)):
                instance_values = [sentence[k+i][feat] for (feat, i) in self.template]
                yield Instance(*instance_values)
        except IndexError:
            pass

    def close(self):
        if hasattr(self._index, 'close'): self._index.close()
        if hasattr(self._sets, 'close'): self._sets.close()
        self._index = self._sets = None

    def _instance_key(self, instance):
        h = 5381  # 5381 is from djb2, https://stackoverflow.com/questions/10696223/reason-for-the-number-5381-in-the-djb-hash-function
        for term in instance:
            for c in term:
                h = h * 65587 + c  # 65587 is from https://github.com/davidar/sdbm/blob/29d5ed2b5297e51125ee45f6efc5541851aab0fb/hash.c#L16
        return h % self._index_size

    def search(self, instance):
        key = self._instance_key(instance)
        # find the start position in the table
        set_start = self._index[key]
        return IndexSet(self._sets, set_start)

    def _sort_file_unique(self, unsorted_tmpfile, sorted_tmpfile, *extra_args):
        subprocess.run(['sort'] + list(extra_args) + ['--unique', '--output', sorted_tmpfile, unsorted_tmpfile])

    def _count_lines_in_file(self, file):
        wc_output = subprocess.run(['wc', '-l', file], capture_output=True).stdout
        return int(wc_output.split()[0])

    def build_index(self, corpus, load_factor=1.0, keep_tmpfiles=False, **unused_kwargs):
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
        log(f" -> sorted {nr_instances} unique instances", self._verbose, start=t0)

        self._index_size = math.ceil(nr_instances / load_factor)

        # Build file of key-sentence pairs
        t0 = time.time()
        nr_lines = 0        
        with open(unsorted_tmpfile, 'w') as TMP:
            for n, sentence in enumerate(corpus.sentences(), 1):   # number sentences from 1
                for instance in self._yield_instances(sentence):
                    key = self._instance_key(instance)
                    if str(instance) == "ART ADJ": thekey = key
                    print(f"{key}\t{n}", file=TMP)
                    nr_lines += 1
        log(f" -> created unsorted key-sentence file, {nr_lines} lines", self._verbose, start=t0)
        t0 = time.time()
        self._sort_file_unique(unsorted_tmpfile, sorted_tmpfile, '--numeric-sort', '--key=1', '--key=2')
        log(f" -> sorted key-sentence pairs", self._verbose, start=t0)

        # Build index file and sets file
        t0 = time.time()
        nr_keys = 0 #nr_elements = 0
        with open(sorted_tmpfile) as TMP:
            current = set_start = set_size = -1
            # Dummy sentence to account for null pointers:
            self._sets.append(0)
            # nr_elements += 1
            for line in TMP:
                key, sent = map(int, line.rsplit(maxsplit=1))
                if current < key:
                    if set_start >= 0:
                        assert current >= 0 and set_size > 0
                        # Now the set is full, and we can write the size of the set at its beginning
                        self._sets[set_start] = set_size
                    for _ in range(current + 1, key):
                        self._index.append(0)
                    # Add a placeholder for the size of the set
                    set_start = len(self._sets)
                    set_size = 0
                    self._sets.append(set_size)
                    self._index.append(set_start)
                    # nr_elements += 1
                    current = key
                    nr_keys += 1
                self._sets.append(sent)
                set_size += 1
                # nr_elements += 1
            # Write the size of the final set at its beginning
            self._sets[set_start] = set_size
            while len(self._index) < self._index_size:
                self._index.append(0)
        log(f" -> created index file with {nr_keys} keys, sets file with {len(self._sets)} elements", self._verbose, start=t0)
        sizes = [BYTES * len(f)/1024/1024 for f in (self._index, self._sets)]
        log(f" -> created .index ({sizes[0]:.1f} mb), .sets ({sizes[1]:.1f} mb)", self._verbose, start=start_time)

        # Cleanup
        if not keep_tmpfiles:
            unsorted_tmpfile.unlink()
            sorted_tmpfile.unlink()
        self.close()
        log("", self._verbose)




################################################################################
## Index set

class IndexSet:
    def __init__(self, setsarray, start):
        self._setsarray = setsarray
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
            result = set()  # []
            selfiter, otheriter = iter(sorted(self)), iter(other)
            selfval, otherval = next(selfiter), next(otheriter)
            while True:
                try:
                    if selfval == otherval:
                        result.add(selfval)  # result.append
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
        self.values = {elem for elem in self if check(elem)}
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


################################################################################
## Corpus

class Corpus:
    def __init__(self, corpus, mode='r', verbose=False):
        basefile = Path(corpus)
        self._corpus = open(basefile.with_suffix('.csv'), 'rb')
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
        while not line or line.startswith(b"# sentence"):
            startpos = corpus.tell()
            line = corpus.readline()
            if not line:
                return None, None
            line = line.strip()
        sentence = []
        while line and not line.startswith(b"# sentence"):
            token = dict(zip(features, line.split(b'\t')))
            sentence.append(token)
            line = corpus.readline().strip()
        return startpos, sentence

    def lookup_sentence(self, n):
        self._index.seek(n * BYTES)
        csv_pos = int.from_bytes(self._index.read(BYTES), byteorder=BYTEORDER)
        self._corpus.seek(csv_pos)
        _pos, sentence = self._get_next_sentence()
        return sentence

    def build_index(self):
        t0 = time.time()
        self.reset()
        self._index.write((0).to_bytes(BYTES, byteorder=BYTEORDER))  # number sentences from 1
        ctr = 0
        while True:
            pos, sentence = self._get_next_sentence()
            if sentence is None:
                break
            self._index.write(pos.to_bytes(BYTES, byteorder=BYTEORDER))
            ctr += 1
        log(f"Built corpus index, {ctr} sentences", self._verbose, start=t0)
        log("", self._verbose)

################################################################################
## Queries

QUEREGEX = re.compile(rb'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    def __init__(self, querystr):
        querystr = _bytesify(querystr)
        querystr = querystr.replace(b' ', b'')
        if not QUEREGEX.match(querystr):
            raise ValueError(f"Error in query: {querystr}")
        tokens = querystr.split(b'][')
        self.query = []
        for tok in tokens:
            self.query.append([])
            parts = re.findall(rb'\w+="[^"]+"', tok)
            for part in parts:
                feat, value = part.split(b'=', 1)
                value = value.replace(b'"', b'')
                self.query[-1].append((feat, value))

    def __str__(self):
        return " ".join("[" + " ".join(f'{feat.decode()}="{val.decode()}"' for feat, val in subq) + "]" 
                        for subq in self.query)

    def subqueries(self):
        # Pairs of tokens
        for i, tok in enumerate(self.query):
            for feat, value in tok:
                for dist in range(1, len(self.query)-i):
                    for feat1, value1 in self.query[i+dist]:
                        templ = Template((feat, 0), (feat1, dist))
                        yield (templ, Instance(value, value1))
        # Single tokens: yield subqueries after more complex queries!
        for tok in self.query:
            for feat, value in tok:
                yield (Template((feat, 0)), Instance(value))

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
        positions = sorted({pos for _, pos in template})
        query = {(feat, pos, val) for ((feat, pos), val) in zip(template, instance)}
        for base in positions:
            subquery = {(feat, base+pos, val) for ((feat, pos), val) in zip(subtemplate, subinstance)}
            if subquery.issubset(query):
                return True
        return False


def query_corpus(args):
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
            index = Index(args.corpus, template)
        except FileNotFoundError:
            continue
        t0 = time.time()
        sentences = index.search(instance)
        log(f"   {index} = {instance} --> {len(sentences)}", args.verbose, start=t0)
        search_results.append((index, instance, sentences))

    log("Sorting:", args.verbose)
    t0 = time.time()
    search_results.sort(key=lambda r: len(r[-1]))
    log("   " + " ".join(str(index) for index, _, _ in search_results), args.verbose, start=t0)

    log("Intersecting:", args.verbose)
    result = None
    for index, instance, sentences in search_results:
        t0 = time.time()
        if result is None:
            result = sentences
        else:
            result.intersection_update(sentences)
        log(f"   {index} = {instance} : {len(sentences)} --> {result}", args.verbose, start=t0)

    if args.filter:
        log("Final filter:", args.verbose)
        t0 = time.time()
        corpus = Corpus(args.corpus)
        result.filter(lambda sent: query.check_sentence(corpus.lookup_sentence(sent)))
        log(f"   {query} --> {result}", args.verbose, start=t0)

    if args.out:
        t0 = time.time()
        with open(args.out, "w") as OUT:
            for sent in sorted(result):
                print(sent, corpus.lookup_sentence(sent), file=OUT)
        log(f"{len(result)} sentences written to {args.out}", args.verbose, start=t0)

    log("", args.verbose)
    log(f"Result: {result}", args.verbose, start=starttime)
    print(result)
    for index, _, _ in search_results:
        index.close()


################################################################################
## Building indexes

def build_indexes(args):
    basedir = args.corpus.with_suffix(Index.dir_suffix)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    corpus = Corpus(args.corpus, mode='w', verbose=args.verbose)
    t0 = time.time()
    corpus.build_index()
    ctr = 1
    for template in yield_templates(args.features, args.max_dist):
        Index(args.corpus, template, mode='w', verbose=args.verbose).build_index(corpus)
        ctr += 1
    log(f"Created {ctr} indexes", args.verbose, start=t0)


def yield_templates(features, max_dist):
    for feat in features:
        yield Template((feat, 0))
        for feat1 in features:
            for dist in range(1, max_dist+1):
                yield Template((feat, 0), (feat1, dist))


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
pgroup = parser.add_mutually_exclusive_group(required=True)
pgroup.add_argument('query', nargs='?', help='the query')
parser.add_argument('--filter', action='store_true', help='filter the final results (might take time)')
parser.add_argument('--out', type=Path, help='file to output the result (one sentence per line)')
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
