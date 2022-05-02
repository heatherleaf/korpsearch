
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
from disk import *
import sqlite3
from dataclasses import dataclass


def log(output, verbose, start=None):
    if verbose:
        if start:
            duration = time.time()-start
            print(output.ljust(100), f"{duration//60:4.0f}:{duration%60:05.2f}")
        else:
            print(output)


################################################################################
## Templates and instances

def _bytesify(s):
    return s.encode() if isinstance(s, str) else bytes(s)


class Template:
    def __init__(self, *feature_positions):
        self._feature_positions = [(_bytesify(feat), pos) for feat, pos in feature_positions]

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
        return b' '.join(map(_bytesify, self._values))

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
    dir_suffix = '.instanceindex'

    def __init__(self, corpus, template, mode='r', verbose=False):
        assert mode in "rw"
        assert isinstance(template, Template)
        self.basedir = corpus.path().with_suffix(self.dir_suffix)
        self.corpus = corpus
        self.template = template
        self._verbose = verbose
        basefile = self._basefile()

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
        start, end = 0, len(self)-1
        while start <= end:
            mid = (start + end) // 2
            key = [keyarray[mid] for keyarray in self._keys]
            instance_key = [str.index for str in instance]
            if key == instance_key:
                return self._index[mid]
            elif key < instance_key:
                start = mid + 1
            else:
                end = mid - 1
        raise ValueError("Key not found")

    def build_index(self, keep_tmpfiles=False, **unused_kwargs):
        log(f"Building index for {self}", self._verbose)
        start_time = t0 = time.time()

        dbfile = self._basefile().with_suffix('.db.tmp')
        con = sqlite3.connect(dbfile)

        # Switch off journalling etc to speed up database creation
        con.execute('pragma synchronous = off')
        con.execute('pragma journal_mode = off')

        # Create table - one column per feature, plus one column for the sentence
        features = ', '.join(f'feature{i}' for i in range(len(self.template)))
        feature_types = ', '.join(f'feature{i} int' for i in range(len(self.template)))
        con.execute(f'''
            create table features(
               {feature_types},
               sentence int,
               primary key ({features}, sentence)
            ) without rowid''')

        # Add all features
        def rows():
            for n, sentence in enumerate(self.corpus.sentences(), 1):
                for instance in self._yield_instances(sentence):
                    yield *(value.index for value in instance.values()), n

        places = ', '.join('?' for _ in range(len(self.template)))
        con.executemany(f'insert or ignore into features values({places}, ?)', rows())

        nr_sentences = self.corpus.num_sentences()
        nr_instances = con.execute(f'select count(*) from (select distinct {features} from features)').fetchone()[0]
        nr_rows = con.execute(f'select count(*) from features').fetchone()[0]
        log(f" -> created instance database, {nr_rows} rows, {nr_instances} instances, {nr_sentences} sentences", self._verbose, start=t0)

        # Build keys files, index file and sets file
        t0 = time.time()
        self._keys = [DiskIntArrayBuilder(path, max_value = len(self.corpus.strings(feat)))
                for (feat, _), path in zip(self.template, self._keypaths)]
        self._sets = DiskIntArrayBuilder(self._setspath, max_value = nr_sentences+1)
        self._index = DiskIntArrayBuilder(self._indexpath, max_value = nr_rows)

        nr_keys = nr_elements = 0
        current = None
        set_start = set_size = -1
        # Dummy sentence to account for null pointers:
        self._sets.append(0)
        nr_elements += 1
        for row in con.execute(f'select * from features order by {features}, sentence'):
            key = row[:-1]
            sent = row[-1]
            if current != key:
                if set_start >= 0:
                    assert set_size > 0
                    # Now the set is full, and we can write the size of the set at its beginning
                    self._sets[set_start] = set_size
                for builder, k in zip(self._keys, key):
                    builder.append(k)
                # Add a placeholder for the size of the set
                set_start, set_size = len(self._sets), 0
                self._sets.append(set_size)
                self._index.append(set_start)
                nr_elements += 1
                current = key
                nr_keys += 1
            self._sets.append(sent)
            set_size += 1
            nr_elements += 1
        # Write the size of the final set at its beginning
        self._sets[set_start] = set_size
        log(f" -> created index file with {nr_keys} keys, sets file with {nr_elements} elements", self._verbose, start=t0)
        log("", self._verbose)

        # Cleanup
        if not keep_tmpfiles:
            dbfile.unlink()
        self.close()




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

def build_corpus_index(corpusfile, verbose=False):
    log(f"Building corpus index", verbose)
    basedir = Path(corpusfile).with_suffix('.corpus')
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    corpus = open(Path(corpusfile).with_suffix('.csv'), 'rb')
    # the first line in the CSV should be a header with the names of each column (=features)
    features = corpus.readline().decode('utf-8').split()

    with open(basedir / 'features', 'w') as features_file:
        json.dump(features, features_file)

    def words():
        # Skip over the first line
        corpus.seek(0)
        corpus.readline()

        new_sentence = True

        while True:
            line = corpus.readline()
            if not line: return

            line = line.strip()
            if line.startswith(b"# sentence"):
                new_sentence = True
            else:
                word = line.split(b'\t')
                while len(word) < len(features):
                    word.append(b'')
                yield new_sentence, word
                new_sentence = False

    t0 = time.time()
    strings = [set() for _feature in features]
    count = 0
    for _new_sentence, word in words():
        count += 1
        for i, feature in enumerate(word):
            strings[i].add(feature)
    log(f" -> read {sum(map(len, strings))} distinct strings", verbose, start=t0)

    t0 = time.time()
    sentence_builder = DiskIntArrayBuilder(basedir / 'sentences',
        max_value = count-1, use_mmap = True)
    feature_builders = []
    for i, feature in enumerate(features):
        path = basedir / ('feature.' + feature)
        builder = DiskStringArrayBuilder(path, strings[i])
        feature_builders.append(builder)

    sentence_builder.append(0) # sentence 0 doesn't exist

    sentence_count = 0
    word_count = 0
    for new_sentence, word in words():
        if new_sentence:
            sentence_builder.append(word_count)
            sentence_count += 1
        for i, feature in enumerate(word):
            feature_builders[i].append(feature)
        word_count += 1

    sentence_builder.close()
    for builder in feature_builders: builder.close()

    log(f" -> built corpus index, {word_count} words, {sentence_count} sentences", verbose, start=t0)
    log("", verbose)

class Corpus:
    def __init__(self, corpus):
        basedir = Path(corpus).with_suffix('.corpus')
        self._path = Path(corpus)
        self._features = json.load(open(basedir / 'features', 'r'))
        self._features = [f.encode('utf-8') for f in self._features]
        self._sentences = DiskIntArray(basedir / 'sentences')
        self._words = \
            {feature: DiskStringArray(basedir / ('feature.' + feature.decode('utf-8')))
             for feature in self._features}
        
    def __str__(self):
        return f"[Corpus: {self._path.stem}]"

    def path(self):
        return self._path

    def strings(self, feature):
        return self._words[feature]._strings

    def intern(self, feature, value):
        return self._words[feature].intern(value)

    def num_sentences(self):
        return len(self._sentences)-1

    def sentences(self):
        for i in range(1, len(self._sentences)):
            yield self.lookup_sentence(i)

    def lookup_sentence(self, n):
        start = self._sentences[n]
        if n+1 < len(self._sentences):
            end = self._sentences[n+1]
        else:
            end = len(self._sentences)

        return [Word(self, i) for i in range(start, end)]

@dataclass(frozen=True)
class Word:
    corpus: Corpus
    pos: int

    def __getitem__(self, feature):
        return self.corpus._words[feature][self.pos]

    def keys(self):
        return self.corpus._features

    def items(self):
        for feature, value in self.corpus._words.items():
            yield feature, value[self.pos]

    def __str__(self):
        return str(dict(self.items()))

    def __repr__(self):
        return f"Word({dict(self.items())})"

    def __eq__(self, other):
        return dict(self) == dict(other)

################################################################################
## Queries

QUEREGEX = re.compile(rb'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    def __init__(self, corpus, querystr):
        self._corpus = corpus
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
                self.query[-1].append((feat, self._corpus.intern(feat, value)))

    def __str__(self):
        return " ".join("[" + " ".join(f'{feat.decode()}="{bytes(val).decode()}"' for feat, val in subq) + "]"
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
    corpus = Corpus(args.corpus)
    query = Query(corpus, args.query)
    starttime = time.time()
    log(f"Query: {args.query} --> {query}", args.verbose)
   
    log("Searching:", args.verbose)
    search_results = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = Index(corpus, template)
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
    t0 = time.time()
    build_corpus_index(args.corpus, verbose=args.verbose)
    corpus = Corpus(args.corpus)
    ctr = 1
    for template in yield_templates(args.features, args.max_dist):
        Index(corpus, template, mode='w', verbose=args.verbose).build_index()
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
