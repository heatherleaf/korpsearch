
import argparse
from pathlib import Path
from index import Index
from corpus import Corpus, render_sentence
from query import Query
from util import setup_logger
import logging


def search_corpus(args):
    corpus = Corpus(args.corpus)
    query = Query(corpus, args.query)
    logging.info(f"Query: {args.query} --> {query}")
   
    logging.debug("Searching...")
    search_results = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = Index(corpus, template)
        except FileNotFoundError:
            continue
        sentences = index.search(instance)
        logging.debug(f"   {index} = {instance} --> {len(sentences)}")
        search_results.append((index, instance, sentences))
    logging.info(f"Searched {len(search_results)} indexes")

    logging.debug("Sorting indexes...")
    search_results.sort(key=lambda r: len(r[-1]))
    logging.debug("   " + " ".join(str(index) for index, _, _ in search_results))

    logging.debug("Intersecting...")
    result = None
    for index, instance, sentences in search_results:
        if result is None:
            result = sentences
        else:
            result.intersection_update(sentences)
        logging.debug(f"   {index} = {instance} : {len(sentences)} --> {result}")
    logging.info(f"After intersection: {result}")

    if args.filter:
        logging.debug("Filtering...")
        result.filter(lambda sent: query.check_sentence(corpus.lookup_sentence(sent)))
        logging.debug(f"   {query} --> {result}")
        logging.info(f"After filtering: {result}")

    if args.out:
        logging.debug("Printing results...")
        with open(args.out, "w") as OUT:
            for sent in sorted(result):
                print(sent, render_sentence(corpus.lookup_sentence(sent)), file=OUT)
        logging.info(f"{len(result)} sentences written to {args.out}")

    logging.info(f"Final result: {result}")
    for index, _, _ in search_results:
        index.close()


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
    args = parser.parse_args()
    setup_logger('{levelname:<6s} {relativeCreated:8.2f} s | {message}', timedivider=1000, loglevel=args.loglevel)
    search_corpus(args)
