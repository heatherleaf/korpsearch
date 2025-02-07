
import hashlib
from pathlib import Path
import logging
import json
import time
from typing import Any, Optional
from argparse import Namespace

from index import Index
from indexset import IndexSet, MergeType
from corpus import Corpus
from query import Query
from util import SENTENCE, WORD

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

INFO_FILE = Path('__info__')


def hash_repr(*objs: object, size: int = 16) -> str:
    hasher = hashlib.md5()
    for obj in objs:
        hasher.update(repr(obj).encode())
    return hasher.hexdigest() [:size]


def hash_query(corpus: Corpus, query: Query, **extra_args: object) -> Path:
    corpus_hash = hash_repr(corpus, size=8)
    query_dir = CACHE_DIR / (corpus.path.stem + '.' + corpus_hash)
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


def run_query(query: Query, results_file: Optional[Path], args: Namespace) -> IndexSet:
    search_results: list[tuple[Query, IndexSet]] = []
    subqueries: list[tuple[Query, Index]] = []
    for subq in query.subqueries():
        if args.no_binary and len(subq) > 1:
            continue
        try:
            subqueries.append((subq, subq.index()))
        except (FileNotFoundError, ValueError):
            continue

    logging.info(f"Searching {len(subqueries)} indexes:")
    maxwidth = max(len(str(subq)) for subq, _ in subqueries)
    for subq, index in subqueries:
        if any(subq.subsumed_by([superq]) for superq, _ in search_results):
            logging.debug(f"     -- subsumed: {subq}")
            continue
        try:
            results = index.search(subq.instance(), offset=subq.min_offset())
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
        logging.debug(f"{i}     {subq!s:{maxwidth}} : {len(results)} elements")

    subq, intersection = search_results[0]
    assert not subq.is_negative()
    logging.info(f"Intersecting {len(search_results)} search results:")
    logging.info(f"     {subq!s:{maxwidth}} = {intersection}")
    used_queries = [subq]
    for subq, results in search_results[1:]:
        if subq.subsumed_by(used_queries):
            logging.debug(f"     -- subsumed: {subq}")
            continue
        intersection_type = intersection.merge_update(
            results,
            results_file,
            use_internal = args.internal_merge,
            merge_type = MergeType.DIFFERENCE if subq.is_negative() else MergeType.INTERSECTION
        )
        logging.info(f" /\\{intersection_type[0].upper()} {subq!s:{maxwidth}} = {intersection}")
        used_queries.append(subq)
        if len(intersection) == 0:
            logging.debug(f"Empty intersection, quitting early")
            break
    return intersection


def search_corpus(corpus: Corpus, query: Query, args: Namespace) -> IndexSet:
    final_results_file = None if args.no_diskarray else hash_query(corpus, query, filtered=args.filter)
    try:
        assert final_results_file and not args.no_cache
        results = IndexSet.open(final_results_file)
        logging.debug(f"Using cached results file: {final_results_file}")
        return results
    except (FileNotFoundError, AssertionError):
        pass

    if not args.filter:
        return run_query(query, final_results_file, args)

    unfiltered_results_file = None if args.no_diskarray else hash_query(corpus, query, filtered=False)
    assert unfiltered_results_file != final_results_file
    try:
        assert unfiltered_results_file and not args.no_cache
        results = IndexSet.open(unfiltered_results_file)
        logging.debug(f"Using cached unfiltered results file: {unfiltered_results_file}")
    except (FileNotFoundError, AssertionError):
        results = run_query(query, unfiltered_results_file, args)
        logging.debug(f"Unfiltered results: {results}")

    results.filter_update(query.check_position, final_results_file)
    return results


def main_search(args: Namespace) -> dict[str, Any]:
    if not (args.end and args.end >= 0):
        args.end = args.start + args.num - 1
    start, end = args.start, args.end
    start_time = time.time()

    corpora: list[str] = args.corpus
    if isinstance(corpora, (str, Path)):
        corpora = [cid.strip() for cid in str(corpora).split(",")]

    corpus_hits: dict[str, int] = {}
    matches: list[dict[str, Any]] = []
    for corpus_id in corpora:
        logging.info(f"Searching in corpus: {corpus_id}")
        with Corpus(corpus_id) as corpus:
            query = Query.parse(corpus, args.query, args.no_sentence_breaks)
            logging.info(f"Query: {query}, {query.min_offset()}")

            if args.show:
                features_to_show = args.show.encode().split(b',')
                for f in features_to_show:
                    if f not in corpus.features:
                        raise ValueError(f"Unknown feature: {f}")
            else:
                features_to_show = [
                    feat for feat in corpus.features
                    if feat in query.features
                    if args.no_sentence_breaks or feat != SENTENCE  # don't show the sentence feature
                ]

            # Always include the 'word' feature, and put it first
            if WORD in corpus.features:
                if WORD in features_to_show:
                    features_to_show.remove(WORD)
                features_to_show.insert(0, WORD)

            results = search_corpus(corpus, query, args)
            corpus_hits[corpus.id] = len(results)
            logging.info(f"Results: {results}")

            if  start < len(results) or end >= 0:
                query_offset = query.max_offset() + 1
                try:
                    for match_pos in results.slice(max(0, start), end+1):
                        sentence = corpus.get_sentence_from_position(match_pos)
                        match_start = match_pos - corpus.sentence_pointers.array[sentence]
                        tokens = [
                            {
                                feat.decode(): strings.interned_string(strings[p])
                                for feat in features_to_show
                                for strings in [corpus.tokens[feat]]
                            }
                            for p in corpus.sentence_positions(sentence)
                        ]

                        matches.append({
                            'corpus': corpus.id,
                            'match': {
                                'start': match_start,
                                'end': match_start + query_offset,
                                'position': match_pos,
                            },
                            'sentence': sentence,
                            'tokens': tokens,
                        })
                except IndexError:
                    pass

            start -= len(results)
            end -= len(results)

    return {
        'time': time.time() - start_time,
        'hits': sum(corpus_hits.values()),
        'corpus_hits': corpus_hits,
        'corpus_order': list(corpus_hits),
        'start': args.start,
        'end': args.start + len(matches),
        'kwic': matches,
    }

