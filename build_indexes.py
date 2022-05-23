
import shutil
import argparse
import itertools
from pathlib import Path
from index import Index, Instance, Template
from disk import DiskIntArrayBuilder
from corpus import Corpus, build_corpus_index_from_csv
from util import setup_logger
import logging
import sqlite3


################################################################################
## Building one index

def build_index(index, keep_tmpfiles=False, min_frequency=None):
    logging.debug(f"Building index for {index}...")

    unary_indexes = None
    if min_frequency and len(index.template) > 1:
        unary_indexes = [Index(index.corpus, Template((feat,0)))
                         for (feat, _pos) in index.template]

    dbfile = index.basefile().with_suffix('.db.tmp')
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
    skipped_instances = 0
    def rows():
        nonlocal skipped_instances
        for n, sentence in enumerate(index.corpus.sentences(), 1):
            for instance in yield_instances(index, sentence):
                if unary_indexes and any(
                            len(unary.search(Instance(val))) < min_frequency 
                            for val, unary in zip(instance, unary_indexes)
                        ):
                    skipped_instances += 1
                    continue
                yield tuple(value.index for value in instance.values()) + (n,)

    places = ', '.join('?' for _ in range(len(index.template)))
    con.executemany(f'insert or ignore into features values({places}, ?)', rows())
    if skipped_instances:
        logging.debug(f"Skipped {skipped_instances} low-frequency instances")

    nr_sentences = index.corpus.num_sentences()
    nr_instances = con.execute(f'select count(*) from (select distinct {features} from features)').fetchone()[0]
    nr_rows = con.execute(f'select count(*) from features').fetchone()[0]
    logging.debug(f" --> created instance database, {nr_rows} rows, {nr_instances} instances, {nr_sentences} sentences")

    # Build keys files, index file and sets file
    index._keys = [DiskIntArrayBuilder(path, max_value = len(index.corpus.strings(feat)))
            for (feat, _), path in zip(index.template, index._keypaths)]
    index._sets = DiskIntArrayBuilder(index._setspath, max_value = nr_sentences+1)
    # nr_rows is the sum of all set sizes, but the .sets file also includes the set sizes,
    # so in some cases we get an OverflowError.
    # This happens e.g. for bnc-20M when building lemma0: nr_rows = 16616400 < 2^24 < 16777216 = nr_rows+nr_sets
    # What we need is max_value = nr_rows + nr_sets; this is a simple hack until we have better solution:
    index._index = DiskIntArrayBuilder(index._indexpath, max_value = nr_rows*2)

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
    logging.info(f"Built index for {index}, with {nr_keys} keys, {nr_elements} set elements")

    # Cleanup
    if not keep_tmpfiles:
        dbfile.unlink()
    index.close()


def yield_instances(index, sentence):
    try:
        for k in range(len(sentence)):
            instance_values = [sentence[k+i][feat] for (feat, i) in index.template]
            yield Instance(*instance_values)
    except IndexError:
        pass


################################################################################
## Building the corpus index and the query indexes

def main(args):
    base = Path(args.corpus)
    corpusdir = base.with_suffix(Corpus.dir_suffix)
    indexdir = base.with_suffix(Index.dir_suffix)

    if args.clean:
        shutil.rmtree(corpusdir, ignore_errors=True)
        shutil.rmtree(indexdir, ignore_errors=True)
        logging.info(f"Removed all indexes")

    if args.corpus_index:
        corpusdir.mkdir(exist_ok=True)
        corpusfile = base.with_suffix('.csv')
        build_corpus_index_from_csv(corpusdir, corpusfile)
        logging.info(f"Created the corpus index")

    if args.features or args.templates:
        corpus = Corpus(base)
        indexdir.mkdir(exist_ok=True)
        ctr = 0
        for template in itertools.chain(
                            yield_templates(args.features, args.max_dist),
                            map(parse_template, args.templates),
                        ):
            index = Index(corpus, template, mode='w')
            build_index(index, keep_tmpfiles=args.keep_tmpfiles, min_frequency=args.min_frequency)
            ctr += 1
        logging.info(f"Created {ctr} query indexes")


def parse_template(template_str):
    template = [tuple(feat_dist.split('.')) for feat_dist in template_str.split('-')]
    try:
        return Template(*[(feat, int(dist)) for (feat, dist) in template])
    except ValueError:
        raise ValueError("Ill-formed template: it should be on the form pos.0 or word.0-pos.2")


def yield_templates(features, max_dist):
    for feat in features:
        yield Template((feat, 0))
    for feat in features:
        for feat1 in features:
            for dist in range(1, max_dist+1):
                yield Template((feat, 0), (feat1, dist))


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, help='corpus base name (i.e. without suffix)')

parser.add_argument('--clean', action='store_true',
    help='remove the corpus index and all query indexes')
parser.add_argument('--corpus-index', '-c', action='store_true',
    help='build the corpus index')
parser.add_argument('--features', '-f', nargs='+', default=[],
    help='build all possible (unary and binary) query indexes for the given features')
parser.add_argument('--templates', '-t', nargs='+', default=[],
    help='build query indexes for the given templates: e.g., pos.0 (unary index), or word.0-pos.2 (binary index)')

parser.add_argument('--max-dist', type=int, default=2, 
                    help='[only with the --features option] max distance between token pairs (default: 2)')
parser.add_argument('--min-frequency', type=int, 
                    help='[only for binary indexes] min unary frequency for all values in a binary instance')

parser.add_argument('--keep-tmpfiles', action='store_true', help='keep temporary files')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, help='debugging output')
parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, help='verbose output')


if __name__ == '__main__':
    args = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} min {warningname}| {message}', timedivider=60*1000, loglevel=args.loglevel)
    main(args)
