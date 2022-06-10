
import argparse
from pathlib import Path
from index import Index, Instance, IndexSet
from corpus import Corpus
from query import Query
from util import setup_logger
from typing import List, Tuple
import logging

# Alternative implementation of filtering, which hopefully is easier to transfer to Cython
from filter_sentences import filter_sentences


def search_corpus(args:argparse.Namespace):
    corpus = Corpus(args.corpus)
    query = Query(corpus, args.query)
    logging.info(f"Query: {args.query} --> {query}")
   
    logging.debug("Searching...")
    search_results : List[Tuple[Index, Instance, IndexSet]] = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = Index(corpus, template)
        except FileNotFoundError:
            continue
        try:
            sentences : IndexSet = index.search(instance)
        except KeyError:
            continue
        logging.debug(f"   {index} = {instance} --> {len(sentences)}")
        search_results.append((index, instance, sentences))
    logging.info(f"Searched {len(search_results)} indexes")

    logging.debug("Sorting indexes...")
    search_results.sort(key=lambda r: len(r[-1]))
    logging.debug("   " + " ".join(str(index) for index, _, _ in search_results))

    logging.debug("Intersecting...")
    result = IndexSet([])
    for index, instance, sentences in search_results:
        if not result:
            result = sentences
        else:
            result.intersection_update(sentences)
        logging.debug(f"   {index} = {instance} : {len(sentences)} --> {result}")
    logging.info(f"After intersection: {result}")

    if args.filter:
        logging.debug("Filtering...")
        filter_sentences(result, query)
        # result.filter(query.check_sentence)
        logging.debug(f"   {query} --> {result}")
        logging.info(f"After filtering: {result}")

    if args.out:
        logging.debug("Printing results...")
        with open(args.out, "w") as OUT:
            for sent in sorted(result):
                print(sent, corpus.render_sentence(sent), file=OUT)
        logging.info(f"{len(result)} sentences written to {args.out}")

    logging.info(f"Final result: {result}")


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
parser.add_argument('query', help='the query')
parser.add_argument('--filter', action='store_true', help='filter the final results (might take time)')
parser.add_argument('--out', type=Path, help='file to output the result (one sentence per line)')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, help='debugging output')
parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, help='verbose output')

if __name__ == '__main__':
    args : argparse.Namespace = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    search_corpus(args)
