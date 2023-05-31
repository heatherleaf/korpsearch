
import argparse
import hashlib
from pathlib import Path
import logging
import json
import time
from typing import List, Tuple

from disk import DiskIntArray
from index import Index
from indexset import IndexSet
from corpus import Corpus
from query import Query
from util import setup_logger

BASE_DIR = Path('private/api-queries')
INFO_FILE = Path('__info__')


def hash_repr(*objs, size=16):
    hasher = hashlib.md5()
    for obj in objs:
        hasher.update(repr(obj).encode())
    return hasher.hexdigest() [:size]


def hash_query(corpus:Corpus, query:Query, **extra_args) -> Path:
    corpus_hash = hash_repr(corpus, size=8)
    query_dir = BASE_DIR / (corpus.path.stem + '.' + corpus_hash)
    if not query_dir.is_dir():
        query_dir.mkdir()
    info_file = query_dir / INFO_FILE
    if not info_file.is_file():
        with open(info_file, 'w') as INFO:
            json.dump({
                'corpus': str(corpus.path),
            }, INFO)
    query_hash = hash_repr(query, extra_args)
    return query_dir / query_hash


def run_query(query:Query, results_file:Path, use_internal:bool=False) -> IndexSet:
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
            results = index.search(subq.instance(), offset=subq.offset())
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
        intersection_type = intersection.intersection_update(
            results, results_file, use_internal=use_internal, difference=subq.is_negative()
        )
        logging.info(f" /\\{intersection_type[0].upper()} {subq!s:{maxwidth}} = {intersection}")
        used_queries.append(subq)
        if len(intersection) == 0:
            logging.debug(f"Empty intersection, quitting early")
            break
    return intersection


def search_corpus(corpus:Corpus, query:Query, args:argparse.Namespace) -> IndexSet:
    unfiltered_results_file = hash_query(corpus, query)
    final_results_file = hash_query(corpus, query, filter=args.filter)

    try:
        assert not args.no_cache
        results = IndexSet(DiskIntArray(final_results_file))
        logging.debug(f"Using cached results file: {final_results_file}")

    except (FileNotFoundError, AssertionError):
        if args.filter:
            assert unfiltered_results_file != final_results_file
            try:
                assert not args.no_cache
                results = IndexSet(DiskIntArray(unfiltered_results_file))
                logging.debug(f"Using cached unfiltered results file: {unfiltered_results_file}")
            except (FileNotFoundError, AssertionError):
                results = run_query(query, unfiltered_results_file, args.internal_intersection)
            logging.debug(f"Unfiltered results: {results}")
            results.filter_update(query.check_position, final_results_file)

        else:
            results = run_query(query, final_results_file, args.internal_intersection)

    return results


def main(corpus:Corpus, args:argparse.Namespace):
    out = {}
    start_time = time.time()

    query = Query.parse(corpus, args.query, args.no_sentence_breaks)
    logging.info(f"Query: {query}, {query.offset()}")

    results = search_corpus(corpus, query, args)
    logging.info(f"Results: {results}")
    out['total-matches'] = len(results)

    if args.print:
        if args.features:
            features_to_show = args.features
        else:
            features_to_show = [
                feat for feat in corpus.features 
                if feat in query.features 
                if args.no_sentence_breaks or feat != corpus.sentence_feature  # don't show the sentence feature
            ]

        match_length = query.max_offset()
        matches = []
        try:
            for match_pos in results.slice(args.start, args.start+args.max):
                sentence = corpus.get_sentence_from_position(match_pos)
                match_start = match_pos - corpus.sentence_pointers[sentence]
                tokens = [
                    {feat: str(corpus.tokens[feat][p]) for feat in features_to_show}
                    for p in corpus.sentence_positions(sentence)
                ]
                
                matches.append({
                    'pos': match_pos,
                    'sentence': sentence,
                    'start': match_start,
                    'length': match_length,
                    'tokens': tokens,
                })
        except IndexError:
            pass
        if matches:
            out['first-match'] = args.start
        out['len-matches'] = len(matches)
        out['matches'] = matches

    out['time'] = time.time() - start_time
    print(json.dumps(out))


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', type=Path, required=True, help='path to compiled corpus')
parser.add_argument('--query', '-q', type=str, required=True, help='the query')

parser.add_argument('--print', '-p', action='store_true', help='print results')
parser.add_argument('--start', '-s', type=int, default=0, help='index of first result (default: 0)')
parser.add_argument('--max', '-m', type=int, default=10, help='max number of results (default: 10)')
parser.add_argument('--features', '-f', type=str, nargs='+', help='features to show (default: the ones in the query)')

parser.add_argument('--no-cache', action="store_true", help="don't use cached queries")
parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO, 
    help='verbose output')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, 
    help='debugging output')

parser.add_argument('--no-sentence-breaks', action='store_true', 
    help="don't care about sentence breaks (default: do care)")
parser.add_argument('--internal-intersection', action='store_true', 
    help='use internal (slow) intersection implementation')
parser.add_argument('--filter', action='store_true', 
    help='filter the final results (should not be necessary, and can take time)')

if __name__ == '__main__':
    args = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    with Corpus(args.corpus) as corpus:
        main(corpus, args)
