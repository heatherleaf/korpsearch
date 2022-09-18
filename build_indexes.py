
import shutil
import argparse
import itertools
from pathlib import Path
from typing import Iterator
import logging

from index import Literal, TemplateLiteral, Template, Index
from corpus import Corpus
from util import setup_logger


################################################################################
## Building the corpus index and the query indexes

def main(args:argparse.Namespace):
    base = Path(args.corpus)
    corpusdir = base.with_suffix(Corpus.dir_suffix)
    indexdir = base.with_suffix(Index.dir_suffix)

    if args.clean:
        shutil.rmtree(corpusdir, ignore_errors=True)
        shutil.rmtree(base.with_suffix(Index.dir_suffix), ignore_errors=True)
        logging.info(f"Removed all indexes")

    if args.corpus_index:
        corpusdir.mkdir(exist_ok=True)
        corpusfile = base.with_suffix('.csv')
        Corpus.build_from_csv(corpusdir, corpusfile)
        logging.info(f"Created the corpus index")

    if args.features or args.templates:
        with Corpus(base) as corpus:
            indexdir.mkdir(exist_ok=True)
            templates = sorted(set(yield_templates(corpus, args)))
            logging.debug(f"Creating indexes: {', '.join(map(str, templates))}")
            for template in templates:
                Index.build(
                    corpus, template, 
                    min_frequency=args.min_frequency, 
                    keep_tmpfiles=args.keep_tmpfiles, 
                    use_sqlite=not args.no_sqlite,
                )
            logging.info(f"Created {len(templates)} query indexes")


def yield_templates(corpus:Corpus, args:argparse.Namespace) -> Iterator[Template]:
    if not args.no_sentence_breaks:
        yield Template([TemplateLiteral(0, corpus.sentence_feature)])
    for tmplstr in args.templates:
        yield Template.parse(corpus, tmplstr)
    for feat in args.features:
        yield Template([TemplateLiteral(0, feat)])
        for feat1 in args.features:
            for dist in range(1, args.max_dist+1):
                template = [TemplateLiteral(0, feat), TemplateLiteral(dist, feat1)]
                if not args.no_sentence_breaks:
                    sfeature = corpus.sentence_feature
                    svalue = corpus.intern(sfeature, corpus.sentence_start_value)
                    literals = [
                        Literal(True, offset, sfeature, svalue)
                        for offset in range(1, dist+1)
                    ]
                    yield Template(template, literals)
                else:
                    yield Template(template)


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
    help='build query indexes for the given templates: e.g., pos:0 (unary index), or word:0+pos:2 (binary index)')

parser.add_argument('--max-dist', type=int, default=2, 
    help='[only with the --features option] max distance between token pairs (default: 2)')
parser.add_argument('--min-frequency', type=int, default=0,
    help='[only for binary indexes] min unary frequency for all values in a binary instance (default: 0)')
parser.add_argument('--no-sentence-breaks', action='store_true', 
    help="[only for binary indexes] don't care about sentence breaks (default: do care)")

parser.add_argument('--verbose', '-v', action='store_const', dest='loglevel', const=logging.DEBUG, default=logging.INFO,
    help='verbose output')
parser.add_argument('--silent', action="store_const", dest='loglevel', const=logging.WARNING, default=logging.INFO,
    help='silent (no output)')

parser.add_argument('--keep-tmpfiles', action='store_true', 
    help='keep temporary database files')
parser.add_argument('--no-sqlite', action='store_true', 
    help="don't use sqlite to build suffix arrays, instead sort using quicksort (probably slower)")


if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} min {warningname}| {message}', timedivider=60*1000, loglevel=args.loglevel)
    main(args)
