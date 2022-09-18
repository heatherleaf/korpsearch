
import argparse
from pathlib import Path
import itertools
import logging
from typing import List, Tuple

from index import Index
from indexset import IndexSet
from corpus import Corpus
from query import Query
from util import setup_logger


def search_corpus(corpus:Corpus, args:argparse.Namespace):
    query = Query.parse(corpus, args.query, args.no_sentence_breaks)
    logging.info(f"Query: {query}")

    logging.debug("Searching:")
    search_results : List[Tuple[Query, IndexSet]]= []
    for subq in query.subqueries():
        try:
            index : Index = subq.index()
        except FileNotFoundError:
            # logging.debug(f"    - index not found: {subq}")
            continue
        if any(subq.subsumed_by([superq]) for superq, _ in search_results):
            logging.debug(f"    - subsumed: {subq}")
            continue
        try:
            results : IndexSet = index.search(subq.instance(), offset=subq.offset())
        except KeyError:
            logging.debug(f"    - not found: {subq} = {subq.instance()} in {index}")
            continue
        search_results.append((subq, results))
        logging.debug(f"    {subq} = {results}")
    logging.info(f"Searched {len(search_results)} indexes: " + ', '.join(str(subq) for subq, _ in search_results))
    search_results.sort(key=lambda r: len(r[-1]))
    if search_results[0][0].is_negative():
        first_positive = [q.is_negative() for q,_ in search_results].index(False)
        first_result = search_results[first_positive]
        del search_results[first_positive]
        search_results.insert(0, first_result)
    logging.debug("Intersection order: " + ', '.join(str(subq) for subq, _ in search_results))

    subq, intersection = search_results[0]
    assert not subq.is_negative()
    logging.debug(f"Intersecting: {subq}")
    used_queries = [subq]
    for subq, results in search_results[1:]:
        if subq.subsumed_by(used_queries):
            logging.debug(f"           - subsumed: {subq}")
            continue
        if subq.is_negative():
            intersection.difference_update(results)
        else:
            intersection.intersection_update(results, use_internal=args.internal_intersection)
        logging.debug(f"           /\\ {subq} = {intersection}")
        used_queries.append(subq)
    logging.info(f"After intersection: {intersection}")

    # if args.filter:
    #     intersection.filter(query.check_position)
    #     logging.info(f"Filtering positions: {intersection}")

    if args.print_pos:
        logging.info(f"Printing {len(intersection)} results...")
        for pos in intersection:
            features_to_show = [feat for feat in corpus.features if feat in query.features]
            sent = corpus.get_sentence_from_token(pos)
            print(f"[{sent}:{pos}]", corpus.render_position(pos, features_to_show))
        return

    sentences = IndexSet([
        sent for sent, _group in itertools.groupby(
            corpus.get_sentence_from_token(pos) for pos in intersection
        )
    ])
    logging.info(f"Converting to sentences: {sentences}")

    if args.filter:
        sentences.filter(query.check_sentence)
        logging.info(f"Filtering sentences: {sentences}")

    if args.print:
        logging.info(f"Printing {len(sentences)} sentences...")
        for sent in sentences:
            features_to_show = [feat for feat in corpus.features if feat in query.features or feat == corpus.sentence_feature]
            print(f"[{sent}]", corpus.render_sentence(sent, features_to_show))
    else:
        print(sentences)


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, 
    help='corpus file in .csv format')
parser.add_argument('query', type=str, 
    help='the query')
parser.add_argument('--filter', '-f', action='store_true', 
    help='filter the final results (might take time, but should not be necessary)')
parser.add_argument('--print', '-p', action='store_true', 
    help='output the result(s), one sentence per line')
parser.add_argument('--print-pos', '-q', action='store_true', 
    help='output the result position(s), one position per line')

parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, 
    help='verbose output')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, 
    help='debugging output')

parser.add_argument('--no-sentence-breaks', action='store_true', 
    help="[only for binary indexes] don't care about sentence breaks (default: do care)")
parser.add_argument('--internal-intersection', action='store_true', 
    help='use internal (slow) intersection implementation')

if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    with Corpus(args.corpus) as corpus:
        search_corpus(corpus, args)
