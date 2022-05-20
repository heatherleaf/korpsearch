
import os
import time
import shutil
import argparse
from pathlib import Path
from index import Index, Instance, Template
from disk import DiskIntArrayBuilder
from corpus import Corpus, build_corpus_index
from util import log
import sqlite3

################################################################################
## Building one index

def build_index(index, keep_tmpfiles=False, **unused_kwargs):
    log(f"Building index for {index}", index._verbose)
    t0 = time.time()

    dbfile = index._basefile().with_suffix('.db.tmp')
    con = sqlite3.connect(dbfile)

    # Switch off journalling etc to speed up database creation
    con.execute('pragma synchronous = off')
    con.execute('pragma journal_mode = off')

    # Create table - one column per feature, plus one column for the sentence
    features = ', '.join(f'feature{i}' for i in range(len(index.template)))
    feature_types = ', '.join(f'feature{i} int' for i in range(len(index.template)))
    con.execute(f'''
        create table features(
            {feature_types},
            sentence int,
            primary key ({features}, sentence)
        ) without rowid''')

    # Add all features
    def rows():
        for n, sentence in enumerate(index.corpus.sentences(), 1):
            for instance in _yield_instances(index, sentence):
                yield tuple(value.index for value in instance.values()) + (n,)

    places = ', '.join('?' for _ in range(len(index.template)))
    con.executemany(f'insert or ignore into features values({places}, ?)', rows())

    nr_sentences = index.corpus.num_sentences()
    nr_instances = con.execute(f'select count(*) from (select distinct {features} from features)').fetchone()[0]
    nr_rows = con.execute(f'select count(*) from features').fetchone()[0]
    log(f" -> created instance database, {nr_rows} rows, {nr_instances} instances, {nr_sentences} sentences", index._verbose, start=t0)

    # Build keys files, index file and sets file
    t0 = time.time()
    index._keys = [DiskIntArrayBuilder(path, max_value = len(index.corpus.strings(feat)))
            for (feat, _), path in zip(index.template, index._keypaths)]
    index._sets = DiskIntArrayBuilder(index._setspath, max_value = nr_sentences+1)
    index._index = DiskIntArrayBuilder(index._indexpath, max_value = nr_rows)

    nr_keys = nr_elements = 0
    current = None
    set_start = set_size = -1
    # Dummy sentence to account for null pointers:
    index._sets.append(0)
    nr_elements += 1
    for row in con.execute(f'select * from features order by {features}, sentence'):
        key = row[:-1]
        sent = row[-1]
        if current != key:
            if set_start >= 0:
                assert set_size > 0
                # Now the set is full, and we can write the size of the set at its beginning
                index._sets[set_start] = set_size
            for builder, k in zip(index._keys, key):
                builder.append(k)
            # Add a placeholder for the size of the set
            set_start, set_size = len(index._sets), 0
            index._sets.append(set_size)
            index._index.append(set_start)
            nr_elements += 1
            current = key
            nr_keys += 1
        index._sets.append(sent)
        set_size += 1
        nr_elements += 1
    # Write the size of the final set at its beginning
    index._sets[set_start] = set_size
    log(f" -> created index file with {nr_keys} keys, sets file with {nr_elements} elements", index._verbose, start=t0)
    log("", index._verbose)

    # Cleanup
    if not keep_tmpfiles:
        dbfile.unlink()
    index.close()


def _yield_instances(index, sentence):
    try:
        for k in range(len(sentence)):
            instance_values = [sentence[k+i][feat] for (feat, i) in index.template]
            yield Instance(*instance_values)
    except IndexError:
        pass


################################################################################
## Building the corpus index, and all query indexes

def build_indexes(args):
    basedir = args.corpus.with_suffix(Index.dir_suffix)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    t0 = time.time()
    build_corpus_index(args.corpus, verbose=args.verbose)
    corpus = Corpus(args.corpus)
    ctr = 1
    for template in yield_templates(args.features, args.max_dist):
        index = Index(corpus, template, mode='w', verbose=args.verbose)
        build_index(index)
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
parser.add_argument('--features', '-f', nargs='+', help='features')
parser.add_argument('--max-dist', type=int, default=2, 
                    help='max distance between token pairs (default: 2)')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')

if __name__ == '__main__':
    args = parser.parse_args()
    build_indexes(args)
