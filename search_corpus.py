
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


def search_corpus(args:argparse.Namespace):
    if args.suffix_array:
        from index import SAIndex as Index
    else:
        from index import Index

    corpus = Corpus(args.corpus)
    query = Query(corpus, args.query)
    logging.info(f"Query: {args.query} --> {query}")

    debug_result = lambda index, instance, results: f"{index.template}[{instance}]={len(results)}"        

    logging.debug("Searching...")
    search_results : List[Tuple[Index, Instance, IndexSet]] = [] 
    for template, instance, offset in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = Index(corpus, template)
        except FileNotFoundError:
            continue
        try:
            results : IndexSet = index.search(instance, offset=offset)
        except KeyError:
            continue
        sresult = (index, instance, results)
        search_results.append(sresult)
        logging.debug(f"   {debug_result(*sresult)} --> {results}")
    logging.info(f"Searched {len(search_results)} indexes")

    logging.debug("Determining intersection order...")
    search_results.sort(key=lambda r: len(r[-1]))
    logging.debug("   --> " + ", ".join(f"{debug_result(*sresult)}" for sresult in search_results))

    logging.debug("Intersecting...")
    intersection = IndexSet([])
    for sresult in search_results:
        index, instance, results = sresult
        if not intersection:
            intersection = results
        else:
            intersection.intersection_update(results)
        logging.debug(f"   {debug_result(*sresult)} --> {intersection}")
    logging.info(f"After intersection: {intersection}")

    if args.suffix_array:
        sentences = IndexSet([
            sent for sent, _group in itertools.groupby(
                corpus.get_sentence_from_token(pos) for pos in intersection
            )
        ])
        logging.debug(f"   to sentences --> {sentences}")
    else:
        sentences = intersection

    if args.filter:
        logging.debug("Filtering...")
        sentences.filter(query.check_sentence)
        logging.debug(f"   {query} --> {sentences}")
        logging.info(f"After filtering: {sentences}")

    logging.info(f"Final result: {sentences}")

    if args.print:
        logging.info(f"Printing {len(sentences)} sentences...")
        for sent in sorted(sentences):
            print(sent, corpus.render_sentence(sent))


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

if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    search_corpus(args)
