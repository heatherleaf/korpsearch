
import hashlib
from pathlib import Path
import logging
import json
import time
from argparse import Namespace
from typing import List, Tuple

from disk import DiskIntArray, DiskIntArrayBuilder
from index import Index, collect_and_sort_positions
from indexset import IndexSet, MergeType
from corpus import Corpus
from query import Query
from util import clean_up

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

INFO_FILE = Path('__info__')


def hash_repr(*objs, size=16):
    hasher = hashlib.md5()
    for obj in objs:
        hasher.update(repr(obj).encode())
    return hasher.hexdigest() [:size]


def hash_query(corpus:Corpus, query:Query, **extra_args) -> Path:
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


def collect_and_sort_prefix(index_view:IndexSet, tmpfile:Path, offset:int=0, bytesize:int=4, sorter:str="tmpfile") -> IndexSet:
    values = index_view.values[index_view.start:(index_view.start+index_view.size)]
    def collector(collect):
        for val in values: collect(val.to_bytes(bytesize, 'big'))
    collect_and_sort_positions(collector, tmpfile, 0, bytesize, bytesize, False, sorter)
    return IndexSet(DiskIntArray(tmpfile), offset = offset)

def run_outer(query:Query, results_file:Path, use_internal:bool=False) -> IndexSet:
    tmp_results = Path("tmp_results")
    union = None
    for disjunct in query.expand():
        partial_query = Query(query.corpus, disjunct)
        partial_results = run_query(partial_query, tmp_results, use_internal)
        if union:
            union.merge_update(
                partial_results,
                None, use_internal=use_internal,
                merge_type=MergeType.UNION
            )
            try: partial_results.values.close()
            except AttributeError: pass
        else: union = partial_results
        try: clean_up(tmp_results, [".ia", ".ia-cfg"])
        except FileNotFoundError: pass
    return union

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
            index.close()
            continue
        try:
            results = index.search(subq.instance(), offset=subq.offset())
        except KeyError:
            logging.debug(f"     -- {subq.instance()} not found: {subq}")
            index.close()
            continue
        search_results.append((subq, results))
        logging.info(f"     {subq!s:{maxwidth}} = {results}")

    search_results.sort(key=lambda r: len(r[-1]))
    prefix_tmp = Path("prefix_tmp")
    if search_results[0][0].is_negative() or any(lit.is_prefix() for lit in search_results[0][0].literals):
        try:
            first_ok = [q.is_negative() or any(lit.is_prefix() for lit in q.literals) for q,_ in search_results].index(False)
            first_result = search_results[first_ok]
        except ValueError:
            first_ok = [any(lit.is_prefix() for lit in q.literals) for q,_ in search_results].index(True)
            first_result = search_results[first_ok]
            first_result = (first_result[0],collect_and_sort_prefix(first_result[1], prefix_tmp, first_result[1].offset))
        del search_results[first_ok]
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
        if any(lit.is_prefix() for lit in subq.literals):
            lengths = sorted([len(res[1]) for res in search_results])
            if len(results) > 0.1 * lengths[1]:
                logging.debug(f"     -- skipping prefix: {subq}")
                continue
            results = collect_and_sort_prefix(results, prefix_tmp, results.offset)
            
        intersection_type = intersection.merge_update(
            results,
            results_file, use_internal=use_internal,
            merge_type=MergeType.DIFFERENCE if subq.is_negative() else MergeType.INTERSECTION
        )
        logging.info(f" /\\{intersection_type[0].upper()} {subq!s:{maxwidth}} = {intersection}")
        used_queries.append(subq)
        try: clean_up(prefix_tmp, [".ia", ".ia-cfg"])
        except FileNotFoundError: pass
        if len(intersection) == 0:
            logging.debug(f"Empty intersection, quitting early")
            break
    for _, results in search_results[1:]:
        results.values.close()
    return intersection


def search_corpus(corpus:Corpus, query:Query, filter_results:bool, 
                  no_cache:bool, internal_intersection:bool) -> IndexSet:
    unfiltered_results_file = hash_query(corpus, query)
    final_results_file = hash_query(corpus, query, filter=filter_results)

    try:
        assert not no_cache
        results = IndexSet(DiskIntArray(final_results_file))
        logging.debug(f"Using cached results file: {final_results_file}")

    except (FileNotFoundError, AssertionError):
        if filter_results:
            assert unfiltered_results_file != final_results_file
            try:
                assert not no_cache
                results = IndexSet(DiskIntArray(unfiltered_results_file))
                logging.debug(f"Using cached unfiltered results file: {unfiltered_results_file}")
            except (FileNotFoundError, AssertionError):
                results = run_outer(query, unfiltered_results_file, internal_intersection)
            logging.debug(f"Unfiltered results: {results}")
            results.filter_update(query.check_position, final_results_file)

        else:
            results = run_outer(query, final_results_file, internal_intersection)

    return results


def main_search(args:Namespace) -> dict:
    if not (args.end and args.end >= 0):
        args.end = args.start + args.num - 1

    with Corpus(args.corpus) as corpus:
        out = {}
        start_time = time.time()

        query = Query.parse(corpus, args.query, args.no_sentence_breaks)
        logging.info(f"Query: {query}, {query.offset()}")

        if args.show:
            features_to_show = args.show.split(',')
            for f in features_to_show:
                if f not in corpus.features:
                    raise ValueError(f"Unknown feature: {f}")
        else:
            features_to_show = [
                feat for feat in corpus.features 
                if feat in query.features 
                if args.no_sentence_breaks or feat != corpus.sentence_feature # don't show the sentence feature
                if not feat.endswith("_rev") # don't show reversed features
            ]

        # Always include the 'word' feature, and put it first
        if 'word' in corpus.features:
            if 'word' in features_to_show:
                features_to_show.remove('word')
            features_to_show.insert(0, 'word')

        results = search_corpus(corpus, query, args.filter, args.no_cache, args.internal_intersection)
        logging.info(f"Results: {results}")
        out['hits'] = len(results)

        query_offset = query.max_offset()
        matches = []
        try:
            for match_pos in results.slice(args.start, args.end+1):
                sentence = corpus.get_sentence_from_position(match_pos)
                match_start = match_pos - corpus.sentence_pointers[sentence]
                tokens = [
                    {feat: str(corpus.tokens[feat][p]) for feat in features_to_show}
                    for p in corpus.sentence_positions(sentence)
                ]

                matches.append({
                    'match': {
                        'start': match_start,
                        'end': match_start + query_offset,
                        'pos': match_pos,
                    },
                    'sentence': sentence,
                    'tokens': tokens,
                })
        except IndexError:
            pass
        if matches:
            out['start'] = args.start
            out['end'] = args.start + len(matches) - 1
        out['kwic'] = matches

        out['time'] = time.time() - start_time
        return out
