
import hashlib
from pathlib import Path
import logging
import json
import time
from typing import Any
from argparse import Namespace

from pyroaring import BitMap

from disk import SymbolList, SymbolRange
from index import Index, BinaryIndex
from corpus import Corpus
from query import Query
from util import add_suffix, Feature, SENTENCE, WORD

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

INFO_FILE = Path('__info__')


def hash_repr(*objs: object, size: int = 32) -> str:
    hasher = hashlib.md5()
    for obj in objs:
        hasher.update(repr(obj).encode())
    return hasher.hexdigest() [:size]


def get_cache_file(query: Query, **extra_args: object) -> Path | None:
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
    return add_suffix(query_dir / query_hash, '.bitmap')


def is_cache_file(cache_file: Path|None) -> bool:
    return isinstance(cache_file, Path) and cache_file.is_relative_to(CACHE_DIR)


def try_open_cache(cache_file: Path|None, args: Namespace) -> BitMap | None:
    try:
        assert cache_file and not args.no_cache
        with open(cache_file, "rb") as file:
            result = BitMap.deserialize(file.read())
        cache_file.touch()
        return result
    except (FileNotFoundError, AssertionError):
        return None


def save_to_cache(cache_file: Path|None, bmap: BitMap) -> None:
    if cache_file:
        with open(cache_file, "wb") as file:
            file.write(bmap.serialize())


def run_outer_query(query: Query, results_file: Path|None, args: Namespace) -> BitMap:
    partial_queries = list(query.expand())
    if len(partial_queries) <= 1:
        return run_inner_query(query, results_file, args)

    union_files = [get_cache_file(q, num=n, outer=query) for n, q in enumerate(partial_queries)]
    union_files[-1] = results_file
    union = None
    for partial_query, union_file in zip(partial_queries, union_files):
        cached_union = try_open_cache(union_file, args)
        if cached_union is not None:
            union = cached_union
            logging.debug(f"Using cached union: {union}")
            continue

        partial_results_file = get_cache_file(partial_query)
        partial_results = try_open_cache(partial_results_file, args)
        if partial_results is not None:
            logging.debug(f"Using cached partial results: {partial_results}")
        else:
            partial_results = run_inner_query(partial_query, partial_results_file, args)

        if union is None:
            union = partial_results
        else:
            logging.info(f"Union: {show_result(union)} \\/ {show_result(partial_results)}")
            union |= partial_results
            save_to_cache(union_file, union)
            logging.info(f"  \\/ = {show_result(union)}")

    assert union is not None
    return union


def run_inner_query(query: Query, results_file: Path|None, args: Namespace) -> BitMap:
    search_results: list[tuple[Query, BitMap]] = []
    subqueries: list[tuple[Query, Index]] = []
    for subq in query.subqueries():
        if args.no_binary and len(subq) > 1:
            continue
        try:
            assert subq.template
            subqueries.append((subq, Index.get(subq.corpus, subq.template)))
        except (FileNotFoundError, ValueError):
            continue

    logging.info(f"Searching {len(subqueries)} indexes:")
    maxwidth = max(len(str(subq)) for subq, _ in subqueries)
    for subq, index in subqueries:
        if any(subq.subsumed_by([superq]) for superq, _ in search_results):
            logging.debug(f"    -- subsumed: {subq}")
            continue
        binary_min_freq = 0
        if isinstance(index, BinaryIndex):
            binary_min_freq = index.getconfig().get("min_frequency", 0)
            if binary_min_freq > 0 and subq.contains_prefix():
                logging.debug(f"    -- skipping: {subq}, because prefix query and min-frequency binary index")
                continue
            if all(isinstance(sym, (SymbolRange, SymbolList)) for sym in subq.instance()):
                logging.debug(f"    -- skipping: {subq}, because binary index and only symbol ranges/lists")
                continue
        results = index.search(subq.instance(), offset=subq.min_offset())
        if results:
            search_results.append((subq, results))
            logging.info(f"    {subq!s:{maxwidth}} = {show_result(results)}")
        elif binary_min_freq > 0:
            logging.debug(f"    -- skipping: {subq}, not found in min-frequency binary index")
            continue
        else:
            logging.debug(f"    -- aborting: {subq}, not found")
            return BitMap()

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
            logging.debug(f"{i:3d} {subq!s:{maxwidth}} : {len(results)} elements")
        logging.info(f"Intersecting {len(search_results)} search results:")

    used_queries: list[Query] = []
    intersection = None
    for subq, results in search_results:
        if subq.subsumed_by(used_queries):
            logging.debug(f"   -- subsumed: {subq}")
            continue

        if intersection is None:
            intersection = results
            logging.info(f"    {subq!s:{maxwidth}} = {show_result(intersection)}")
        elif subq.is_negative():
            intersection -= results
            logging.info(f" -- {subq!s:{maxwidth}} = {show_result(intersection)}")
        else:
            intersection &= results
            logging.info(f" /\\ {subq!s:{maxwidth}} = {show_result(intersection)}")

        used_queries.append(subq)
        if len(intersection) == 0:
            logging.debug(f"Empty intersection, quitting early {show_result(intersection)}")
            break

    assert intersection is not None
    save_to_cache(results_file, intersection)
    return intersection


def run_query(query: Query, results_file: Path|None, args: Namespace) -> BitMap:
    return run_outer_query(query, results_file, args)


def search_corpus(query: Query, args: Namespace) -> BitMap:
    final_results_file = get_cache_file(query, filtered=args.filter)
    cached_results = try_open_cache(final_results_file, args)
    if cached_results is not None:
        logging.debug(f"Using cached results file: {final_results_file}")
        return cached_results

    if not args.filter:
        return run_query(query, final_results_file, args)

    unfiltered_results_file = get_cache_file(query, filtered=False)
    assert unfiltered_results_file != final_results_file
    unfiltered_results = try_open_cache(unfiltered_results_file, args)
    if unfiltered_results is not None:
        logging.debug(f"Using cached unfiltered results file: {unfiltered_results_file}")
    else:
        unfiltered_results = run_query(query, unfiltered_results_file, args)
        logging.debug(f"Unfiltered results: {show_result(unfiltered_results)}")

    filtered_results = BitMap(val for val in unfiltered_results if query.check_position(val))
    save_to_cache(final_results_file, filtered_results)
    return filtered_results


def show_result(bmap: BitMap) -> str:
    if not bmap:
        return "{}#0"
    elif len(bmap) < 5:
        return f"{set(bmap)}#{len(bmap)}"
    else:
        return f"{{{bmap[0]}, {bmap[1]}, ..., {bmap[-1]}}}#{len(bmap)}"


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
            logging.info(f"Results: {show_result(results)}")

            if start < len(results) and end >= 0:
                query_offset = query.max_offset() + 1
                try:
                    for match_pos in results[max(0, start) : end+1]:
                        sentence = corpus.get_sentence_from_position(match_pos)
                        match_start = match_pos - corpus.sentence_pointers.array[sentence]
                        tokens = [
                            {
                                feat.decode(): symbol_array.symbols.to_name(symbol_array[p]).decode()
                                for feat in features_to_show
                                for symbol_array in [corpus.tokens[feat]]
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

