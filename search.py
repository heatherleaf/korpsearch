
import hashlib
from pathlib import Path
import logging
import json
import time
from typing import Any
from argparse import Namespace

from index import Index, BinaryIndex
from indexset import IndexSet, MergeType
from corpus import Corpus
from query import Query
from util import Feature, SENTENCE, WORD
from disk import DiskIntArray

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

INFO_FILE = Path('__info__')


def hash_repr(*objs: object, size: int = 32) -> str:
    hasher = hashlib.md5()
    for obj in objs:
        hasher.update(repr(obj).encode())
    return hasher.hexdigest() [:size]


def get_cache_file(args: Namespace, query: Query, **extra_args: object) -> Path | None:
    if args.no_diskarray:
        return None
    corpus_hash = hash_repr(query.corpus, size=16)
    query_dir = CACHE_DIR / f"{query.corpus.path.stem}.{corpus_hash}"
    if not query_dir.is_dir():
        query_dir.mkdir()
    info_file = query_dir / INFO_FILE
    if not info_file.is_file():
        with open(info_file, 'w') as INFO:
            print(json.dumps({
                'name': query.corpus.name,
                'id': query.corpus.id,
            }), file=INFO)
    query_hash = hash_repr(query, extra_args, size=32)
    return DiskIntArray.getpath(query_dir / query_hash)


def is_cache_file(cache_file: Path|None) -> bool:
    return isinstance(cache_file, Path) and cache_file.is_relative_to(CACHE_DIR)


def try_open_cache(cache_file: Path|None, args: Namespace) -> IndexSet | None:
    try:
        assert cache_file and not args.no_cache
        result = IndexSet.open(cache_file)
        cache_file.touch()
        DiskIntArray.getconfig(cache_file).touch()
        return result
    except (FileNotFoundError, AssertionError):
        return None


def collect_and_sort_prefix(index_view: IndexSet, sorted_file: Path|None, args: Namespace) -> IndexSet:
    try:
        assert not args.internal_merge
        from fast_merge import sort  # type: ignore
    except (ModuleNotFoundError, AssertionError):
        from merge import sort
    result = DiskIntArray.create(index_view.size, sorted_file)
    sort(index_view.values.array, index_view.start, index_view.size, result.array)
    return IndexSet(result, path = sorted_file, offset = index_view.offset)


def run_outer_query(query: Query, results_file: Path|None, args: Namespace) -> IndexSet:
    partial_queries = list(query.expand())
    if len(partial_queries) <= 1:
        return run_inner_query(query, results_file, args)

    union_files = [get_cache_file(args, q, num=n, outer=query) for n, q in enumerate(partial_queries)]
    union_files[-1] = results_file
    union = None
    for partial_query, union_file in zip(partial_queries, union_files):
        cached_union = try_open_cache(union_file, args)
        if cached_union is not None:
            union = cached_union
            logging.debug(f"Using cached union: {union}")
            continue

        partial_results_file = get_cache_file(args, partial_query)
        partial_results = try_open_cache(partial_results_file, args)
        if partial_results is not None:
            logging.debug(f"Using cached partial results: {partial_results}")
        else:
            partial_results = run_inner_query(partial_query, partial_results_file, args)

        if union is None:
            union = partial_results
        else:
            logging.info(f"Union: {union} \\/ {partial_results}")
            union_type = union.merge_update(
                partial_results,
                union_file,
                use_internal = args.internal_merge,
                merge_type = MergeType.UNION,
            )
            logging.info(f"  \\/{union_type[0].upper()}  = {union}")
            if partial_results.path != union.path:
                partial_results.values.close()

    assert union is not None
    return union


def run_inner_query(query: Query, results_file: Path|None, args: Namespace) -> IndexSet:
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
        if isinstance(index, BinaryIndex) and subq.contains_prefix():
            if index.getconfig().get("min_frequency", 0) > 0:
                logging.debug(f"     -- skipping: {subq}, because prefix query and min-frequency binary index")
                continue
        try:
            results = index.search(subq.instance(), offset=subq.min_offset())
        except KeyError:
            logging.debug(f"     -- not found: {subq}")
            continue
        search_results.append((subq, results))
        logging.info(f"     {subq!s:{maxwidth}} = {results}")

    search_results.sort(key=lambda r: len(r[-1]))
    first_query = search_results[0][0]
    if first_query.is_negative() or first_query.contains_prefix():
        try:
            first_ok = [q.is_negative() or q.contains_prefix() for q,_ in search_results].index(False)
        except ValueError:
            first_ok = [q.contains_prefix() for q,_ in search_results].index(True)
        first_result = search_results[first_ok]
        del search_results[first_ok]
        search_results.insert(0, first_result)
    assert not search_results[0][0].is_negative()

    if len(search_results) > 1:
        logging.debug("Intersection order:")
        for i, (subq, results) in enumerate(search_results, 1):
            logging.debug(f"{i}     {subq!s:{maxwidth}} : {len(results)} elements")
        logging.info(f"Intersecting {len(search_results)} search results:")

    used_queries: list[Query] = []
    intersection = None
    for subq, results in search_results:
        if subq.subsumed_by(used_queries):
            logging.debug(f"     -- subsumed: {subq}")
            continue
        if subq.contains_prefix():
            sorted_file = get_cache_file(args, subq, sorted_prefix=True)
            sorted_results = try_open_cache(sorted_file, args)
            if sorted_results is not None:
                spec = "cached sorted"
            else:
                sorted_results = collect_and_sort_prefix(results, sorted_file, args)
                spec = "sorted"
            results = sorted_results
            logging.debug(f"     -- {spec} prefix query: {subq} = {results}")

        if intersection is None:
            intersection = results
            logging.info(f"     {subq!s:{maxwidth}} = {intersection}")
        else:
            intersection_type = intersection.merge_update(
                results,
                results_file,
                use_internal = args.internal_merge,
                merge_type = MergeType.DIFFERENCE if subq.is_negative() else MergeType.INTERSECTION
            )
            logging.info(f" /\\{intersection_type[0].upper()} {subq!s:{maxwidth}} = {intersection}")

        used_queries.append(subq)
        if len(intersection) == 0:
            logging.debug(f"Empty intersection, quitting early {intersection}")
            break

    assert intersection is not None
    for _, results in search_results:
        if results.path != intersection.path:
            results.values.close()
    return intersection


def run_query(query: Query, results_file: Path|None, args: Namespace) -> IndexSet:
    result = run_outer_query(query, results_file, args)
    if is_cache_file(result.path) and result.path != results_file:
        assert result.path is not None and results_file is not None
        DiskIntArray.getconfig(result.path).replace(DiskIntArray.getconfig(results_file))
        result.path.replace(results_file)
        result.path = results_file
    return result


def search_corpus(query: Query, args: Namespace) -> IndexSet:
    final_results_file = get_cache_file(args, query, filtered=args.filter)
    cached_results = try_open_cache(final_results_file, args)
    if cached_results is not None:
        logging.debug(f"Using cached results file: {final_results_file}")
        return cached_results

    if not args.filter:
        return run_query(query, final_results_file, args)

    unfiltered_results_file = get_cache_file(args, query, filtered=False)
    assert unfiltered_results_file != final_results_file
    unfiltered_results = try_open_cache(unfiltered_results_file, args)
    if unfiltered_results is not None:
        logging.debug(f"Using cached unfiltered results file: {unfiltered_results_file}")
    else:
        unfiltered_results = run_query(query, unfiltered_results_file, args)
        logging.debug(f"Unfiltered results: {unfiltered_results}")

    unfiltered_results.filter_update(query.check_position, final_results_file)
    return unfiltered_results


def main_search(args: Namespace) -> dict[str, Any]:
    if not (args.end and args.end > 0):
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
        with Corpus(corpus_id, base_dir=args.base_dir) as corpus:
            try:
                query = Query.parse(corpus, args.query, args.no_sentence_breaks)
            except ValueError as err:
                logging.info(f"Couldn't parse query {args.query}: {err}")
                continue
            logging.info(f"Query: {query}")

            if args.show:
                features_to_show = [Feature(f) for f in args.show.encode().split(b',')]
                for f in features_to_show:
                    if f not in corpus.features():
                        raise ValueError(f"Unknown feature: {f!r}")
            else:
                features_to_show = [
                    feat for feat in corpus.features()
                    if feat in query.features
                    if args.no_sentence_breaks or feat != SENTENCE  # don't show the sentence feature
                    if not feat.endswith(b'_rev')  # don't show reversed features
                ]

            # Always include the 'word' feature, and put it first
            if WORD in corpus.features():
                if WORD in features_to_show:
                    features_to_show.remove(WORD)
                features_to_show.insert(0, WORD)

            results = search_corpus(query, args)
            corpus_hits[corpus.name] = len(results)
            logging.info(f"Results: {results}")

            if start < len(results) and end >= 0:
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
                            'corpus': corpus.name,
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

