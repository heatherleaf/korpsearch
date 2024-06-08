
import shutil
import argparse
from pathlib import Path
from collections.abc import Iterator
import logging

from index import Literal, TemplateLiteral, Template, Index, SORTER_CHOICES, PIVOT_SELECTORS
from corpus import Corpus
from util import setup_logger, add_suffix, CompressedFileReader


CSV_SUFFIXES = ".csv .tsv .txt .gz .bz2 .xz".split()

################################################################################
## Building the corpus index and the query indexes

def main(args: argparse.Namespace) -> None:
    base = corpusfile = Path(args.corpus)
    while base.suffix in CSV_SUFFIXES: 
        base = base.with_suffix('')
    corpusdir = add_suffix(base, Corpus.dir_suffix)
    indexdir = add_suffix(base, Index.dir_suffix)

    if args.clean:
        shutil.rmtree(corpusdir, ignore_errors=True)
        shutil.rmtree(indexdir, ignore_errors=True)
        logging.info(f"Removed all indexes")

    if args.corpus_index:
        with CompressedFileReader(corpusfile) as _: 
            # This is just to test that the corpus file actually exists
            pass
        corpusdir.mkdir(exist_ok=True)
        try:
            with Corpus(base) as _corpus:
                assert not args.force
            logging.info(f"Corpus index already exists")
        except (FileNotFoundError, AssertionError):
            Corpus.build_from_csv(corpusdir, corpusfile)
            logging.info(f"Created the corpus index")

    if args.features or args.templates:
        with Corpus(base) as corpus:
            indexdir.mkdir(exist_ok=True)
            templates = set(yield_templates(corpus, args))
            existing_templates: set[Template] = set()
            for tmpl in templates:
                try:
                    with Index.get(corpus, tmpl) as _index:
                        assert not args.force
                    existing_templates.add(tmpl)
                except (FileNotFoundError, AssertionError):
                    pass
            if existing_templates:
                logging.info(f"Skipping {len(existing_templates)} existing indexes")
                templates -= existing_templates
            if templates:
                sorted_templates = sorted(templates)
                logging.debug(f"Creating {len(templates)} indexes: {', '.join(map(str, sorted_templates))}")
                for template in sorted_templates:
                    Index.build(corpus, template, args)
                logging.info(f"Created {len(templates)} query indexes")


def yield_templates(corpus: Corpus, args: argparse.Namespace) -> Iterator[Template]:
    sfeature = corpus.sentence_feature
    svalue = corpus.intern(sfeature, corpus.sentence_start_value)
    if args.templates:
        for tmplstr in args.templates:
            tmpl = Template.parse(corpus, tmplstr)
            if args.no_sentence_breaks:
                yield tmpl
            else:
                dist = tmpl.maxdelta()
                literals = set(tmpl.literals) | {
                    Literal(True, offset, sfeature, svalue)
                    for offset in range(1, dist+1)
                }
                yield Template(tmpl.template, literals)
    if args.features:
        if not args.no_sentence_breaks:
            yield Template([TemplateLiteral(0, corpus.sentence_feature)])
        for feat in args.features:
            yield Template([TemplateLiteral(0, feat)])
            for feat1 in args.features:
                for dist in range(1, args.max_dist+1):
                    template = [TemplateLiteral(0, feat), TemplateLiteral(dist, feat1)]
                    if args.no_sentence_breaks:
                        yield Template(template)
                    else:
                        literals = {
                            Literal(True, offset, sfeature, svalue)
                            for offset in range(1, dist+1)
                        }
                        yield Template(template, literals)


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', type=Path, metavar='PATH', required=True, 
    help='corpus base name (i.e. without suffix)')
parser.add_argument('--clean', action='store_true', 
    help='remove the corpus index and all query indexes')
parser.add_argument('--force', action='store_true', 
    help='build indexes even if they exist')
parser.add_argument('--corpus-index', '-i', action='store_true', 
    help='build the corpus index')
parser.add_argument('--features', '-f', nargs='+', default=[], metavar='FEAT',
    help='build all possible (unary and binary) query indexes for the given features')
parser.add_argument('--templates', '-t', nargs='+', default=[], metavar='TMPL',
    help='build query indexes for the given templates: e.g., pos:0 (unary index), or word:0+pos:2 (binary index)')

parser.add_argument('--max-dist', type=int, default=2, metavar='DIST',
    help='[only with the --features option] max distance between token pairs (default: 2)')
parser.add_argument('--min-frequency', type=int, default=0, metavar='FREQ',
    help='[only for binary indexes] min unary frequency for all values in a binary instance (default: 0)')
parser.add_argument('--no-sentence-breaks', action='store_true', 
    help="[only for binary indexes] don't care about sentence breaks (default: do care)")

parser.add_argument('--verbose', '-v', action='store_const', dest='loglevel', const=logging.DEBUG, default=logging.INFO,
    help='verbose output')
parser.add_argument('--silent', action="store_const", dest='loglevel', const=logging.WARNING, default=logging.INFO,
    help='silent (no output)')

parser.add_argument('--sorter', '-s', choices=SORTER_CHOICES, default=SORTER_CHOICES[0],
    help=f'sorter to use: one of {", ".join(SORTER_CHOICES)} (default: {SORTER_CHOICES[0]})')
parser.add_argument('--cutoff', type=int, default=1_000_000, 
    help="[only for sorters 'tmpfile' and 'java'] " 
         "the cutoff when to use the builtin sorting implementation (default: 1 million)")
parser.add_argument('--pivot-selector', choices=PIVOT_SELECTORS, default=next(iter(PIVOT_SELECTORS)), 
    help="[only for sorters 'tmpfile' and 'java'] "
         f"pivot selector: one of {', '.join(PIVOT_SELECTORS)} (default: {next(iter(PIVOT_SELECTORS))})")
parser.add_argument('--keep-tmpfiles', action='store_true', 
    help="keep temporary files (default: don't keep)")


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} min {warningname}| {message}', timedivider=60*1000, loglevel=args.loglevel)
    main(args)
