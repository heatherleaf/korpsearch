
import argparse
from pathlib import Path
import itertools
import logging
from typing import List, Tuple

from disk import DiskIntArray, DiskIntArrayBuilder
from index import Index
from indexset import IndexSet
from corpus import Corpus
from query import Query
from util import setup_logger


def search_corpus(corpus:Corpus, args:argparse.Namespace):
    query = Query.parse(corpus, args.query, args.no_sentence_breaks)
    logging.info(f"Query: {query}")

    search_results : List[Tuple[Query, IndexSet]]= []
    subqueries : List[Tuple[Query, Index]] = []
    for subq in query.subqueries():
        try:
            subqueries.append((subq, subq.index()))
        except FileNotFoundError:
            continue

    logging.info(f"Searching {len(subqueries)} indexes:")
    maxwidth = max(len(str(subq)) for subq, _ in subqueries)
    for subq, index in subqueries:
        if any(subq.subsumed_by([superq]) for superq, _ in search_results):
            logging.debug(f"     -- subsumed: {subq}")
            continue
        try:
            results : IndexSet = index.search(subq.instance(), offset=subq.offset())
        except KeyError:
            logging.debug(f"     -- {subq.instance()} not found: {subq}")
            continue
        search_results.append((subq, results))
        logging.info(f"     {subq!s:{maxwidth}} = {results}")

    search_results.sort(key=lambda r: len(r[-1]))
    if search_results[0][0].is_negative():
        first_positive = [q.is_negative() for q,_ in search_results].index(False)
        first_result = search_results[first_positive]
        del search_results[first_positive]
        search_results.insert(0, first_result)
    logging.debug("Intersection order:")
    for i, (subq, results) in enumerate(search_results, 1):
        logging.debug(f"     {subq!s:{maxwidth}} : {len(results)} elements")

    subq, intersection = search_results[0]
    assert not subq.is_negative()
    logging.info(f"Intersecting {len(search_results)} search results:")
    logging.info(f"     {subq!s:{maxwidth}} = {intersection}")
    used_queries = [subq]
    for subq, results in search_results[1:]:
        if subq.subsumed_by(used_queries):
            logging.debug(f"     -- subsumed: {subq}")
            continue
        intersection_type = intersection.intersection_update(results, args.file, use_internal=args.internal_intersection, difference=subq.is_negative())
        logging.info(f" /\\{intersection_type[0].upper()} {subq!s:{maxwidth}} = {intersection}")
        used_queries.append(subq)

    if args.filter:
        intersection.filter_update(query.check_position, args.file)
        logging.info(f"Filtering positions: {intersection}")

    if args.sentences:
        sentences = [sent for sent, _group in itertools.groupby(
                        corpus.get_sentence_from_position(pos) for pos in intersection
                    )]
        if args.file:
            DiskIntArrayBuilder.build(args.file, sentences)
            sentences = DiskIntArray(args.file)
        intersection = IndexSet(sentences)
        logging.info(f"Converting to sentences: {intersection}")

        if args.filter:
            intersection.filter_update(query.check_sentence, args.file)
            logging.info(f"Filtering sentences: {intersection}")

    logging.debug(f"Result file: {intersection.path}")

    if not args.print:
        print(intersection)
    else:
        logging.info(f"Printing {len(intersection)} results")
        for result in intersection:
            if args.features:
                features_to_show = args.features
            else:
                features_to_show = [
                    feat for feat in corpus.features 
                    if feat in query.features 
                    if args.no_sentence_breaks or feat != corpus.sentence_feature  # don't show the sentence feature
                ]
            if args.sentences:
                print(f"[{result}]", corpus.render_sentence(
                    result, 
                    features=features_to_show,
                ))
            else:
                sent = corpus.get_sentence_from_position(result)
                print(f"[{sent}:{result}]", corpus.render_sentence(
                    sent, 
                    pos=result, 
                    offset=query.max_offset(), 
                    features=features_to_show,
                    context=args.context,
                ))


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, 
    help='corpus file in .csv format')
parser.add_argument('query', type=str, 
    help='the query')

parser.add_argument('--file', '-f', type=Path,
    help='store the array of results in a file')
parser.add_argument('--print', '-p', action='store_true', 
    help='output the result(s), one per line')
parser.add_argument('--context', type=int, default=-1,
    help='context window to print (default: print the whole sentence)')
parser.add_argument('--features', type=str, nargs='+', 
    help='features to print (default: the ones in the query)')

parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, 
    help='verbose output')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, 
    help='debugging output')

parser.add_argument('--no-sentence-breaks', action='store_true', 
    help="[only for binary indexes] don't care about sentence breaks (default: do care)")
parser.add_argument('--internal-intersection', action='store_true', 
    help='use internal (slow) intersection implementation')
parser.add_argument('--filter', action='store_true', 
    help='filter the final results (should not be necessary, and can take time)')
parser.add_argument('--sentences', action='store_true', 
    help='convert the matches to sentences (should not be necessary)')

if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    with Corpus(args.corpus) as corpus:
        search_corpus(corpus, args)
