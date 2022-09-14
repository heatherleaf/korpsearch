
import argparse
from pathlib import Path
from typing import List, Tuple
import itertools
import logging

from index import Instance
from indexset import IndexSet
from corpus import Corpus
from query import Query
from util import setup_logger


def search_corpus(corpus:Corpus, args:argparse.Namespace):
    if args.suffix_array:
        from index import SAIndex as Index
    else:
        from index import Index

    query = Query(corpus, args.query)
    logging.info(f"Query: {query}")

    debug_query = lambda index, instance, offset, positive: \
        ('' if positive else '!') + f"{index.template}[{instance}]-{offset}"

    logging.debug("Searching:")
    search_results : List[Tuple[Index, Instance, int, bool, IndexSet]] = []
    for template, instance, offset, positive in query.subqueries():
        try:
            index = Index(corpus, template)
        except FileNotFoundError:
            continue
        if index.dir_suffix == '.indexes':
            assert positive, "only --suffix-array can use negative queries"
        if any(Query.is_subquery(
                    index.template, instance, offset, positive,
                    prev_index.template, prev_instance, prev_offset, prev_positive,
                )
                for (prev_index, prev_instance, prev_offset, prev_positive, _) in search_results
            ):
            continue
        try:
            results : IndexSet = index.search(instance, offset=offset)
        except KeyError:
            continue
        search_results.append((index, instance, offset, positive, results))
        logging.debug(f"   {debug_query(index, instance, offset, positive)} = {results}")
    logging.info(f"Searched {len(search_results)} indexes: " + ', '.join(
                    f"{debug_query(index, instance, offset, positive)}" 
                    for index, instance, offset, positive, _ in search_results
                ))

    search_results.sort(key=lambda r: len(r[-1]))
    if len(search_results) > 1 and not search_results[0][-2]:
        search_results[:2] = reversed(search_results[:2])
    logging.debug("Intersection order: " + ', '.join(
                    f"{debug_query(index, instance, offset, positive)}" 
                    for index, instance, offset, positive, _ in search_results
                ))

    index, instance, offset, positive, intersection = search_results[0]
    assert positive
    logging.debug(f"Intersecting: {debug_query(index, instance, offset, positive)}")
    for index, instance, offset, positive, results in search_results[1:]:
        # TODO: check for subsumption
        # e.g.: if we have intersected pos:0+pos:1 and pos:1+pos:2, 
        # then we don't need to intersect with pos:0+pos:2
        if positive:
            intersection.intersection_update(results, use_internal=args.internal_intersection)
        else:
            intersection.difference_update(results)
        logging.debug(f"           /\\ {debug_query(index, instance, offset, positive)} = {intersection}")
    logging.info(f"After intersection: {intersection}")

    if args.suffix_array:
        sentences = IndexSet([
            sent for sent, _group in itertools.groupby(
                corpus.get_sentence_from_token(pos) for pos in intersection
            )
        ])
        logging.info(f"Converting to sentences: {sentences}")
    else:
        sentences = intersection

    if args.filter:
        sentences.filter(query.check_sentence)
        logging.info(f"Filtering sentences: {sentences}")

    logging.info(f"Final result: {sentences}")

    if args.print:
        logging.info(f"Printing {len(sentences)} sentences...")
        for sent in sorted(sentences):
            features_to_show = [feat for feat in corpus.features if feat in query.features]
            print(f"[{sent}]", corpus.render_sentence(sent, features_to_show))


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
parser.add_argument('query', type=str, help='the query')
parser.add_argument('--filter', '-f', action='store_true', help='filter the final results (might take time)')
parser.add_argument('--print', '-p', action='store_true', help='output the result(s), one sentence per line')

parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, help='verbose output')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, help='debugging output')

parser.add_argument('--suffix-array', action='store_true', help='use suffix arrays as indexes (experimental)')
parser.add_argument('--internal-intersection', action='store_true', help='use internal (slow) intersection implementation')

if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    with Corpus(args.corpus) as corpus:
        search_corpus(corpus, args)
