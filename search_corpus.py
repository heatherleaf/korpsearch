
import time
import argparse
from pathlib import Path
from index import Index
from corpus import Corpus, render_sentence
from query import Query
from util import log


def search_corpus(args):
    corpus = Corpus(args.corpus)
    query = Query(corpus, args.query)
    starttime = time.time()
    log(f"Query: {args.query} --> {query}", args.verbose)
   
    log("Searching:", args.verbose)
    search_results = []
    for template, instance in query.subqueries():
        if any(Query.is_subquery(template, instance, prev_index.template, prev_instance)
               for (prev_index, prev_instance, _) in search_results):
            continue
        try:
            index = Index(corpus, template)
        except FileNotFoundError:
            continue
        t0 = time.time()
        sentences = index.search(instance)
        log(f"   {index} = {instance} --> {len(sentences)}", args.verbose, start=t0)
        search_results.append((index, instance, sentences))

    log("Sorting:", args.verbose)
    t0 = time.time()
    search_results.sort(key=lambda r: len(r[-1]))
    log("   " + " ".join(str(index) for index, _, _ in search_results), args.verbose, start=t0)

    log("Intersecting:", args.verbose)
    result = None
    for index, instance, sentences in search_results:
        t0 = time.time()
        if result is None:
            result = sentences
        else:
            result.intersection_update(sentences)
        log(f"   {index} = {instance} : {len(sentences)} --> {result}", args.verbose, start=t0)

    if args.filter:
        log("Final filter:", args.verbose)
        t0 = time.time()
        result.filter(lambda sent: query.check_sentence(corpus.lookup_sentence(sent)))
        log(f"   {query} --> {result}", args.verbose, start=t0)

    if args.out:
        t0 = time.time()
        with open(args.out, "w") as OUT:
            for sent in sorted(result):
                print(sent, render_sentence(corpus.lookup_sentence(sent)), file=OUT)
        log(f"{len(result)} sentences written to {args.out}", args.verbose, start=t0)

    log("", args.verbose)
    log(f"Result: {result}", args.verbose, start=starttime)
    print(result)
    for index, _, _ in search_results:
        index.close()


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('corpus', type=Path, help='corpus file in .csv format')
parser.add_argument('query', help='the query')
parser.add_argument('--filter', action='store_true', help='filter the final results (might take time)')
parser.add_argument('--out', type=Path, help='file to output the result (one sentence per line)')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')

if __name__ == '__main__':
    args = parser.parse_args()
    search_corpus(args)
