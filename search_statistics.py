
import argparse
from pathlib import Path
import logging
import time
from typing import Any
from argparse import Namespace
from collections import Counter

from corpus import Corpus
from query import Query
from util import Feature
from util import setup_logger
from search import search_corpus


def main_count(args: Namespace) -> dict[str, Any]:
    start_time = time.time()
    gfeat = Feature(args.group_by.encode("ascii"))

    corpora: list[str] = args.corpus
    if isinstance(corpora, (str, Path)):
        corpora = [cid.strip() for cid in str(corpora).split(",")]

    total_tokens = 0
    total_stats: Counter[tuple[str, ...]] = Counter()
    corpus_stats: dict[str, Any] = {}
    sums: dict[str, int]  # this is just an approximation: sums["relative"] is a float
    for corpus_id in corpora:
        logging.info(f"Searching in corpus: {corpus_id}")
        with Corpus(corpus_id, base_dir=args.base_dir) as corpus:
            try:
                query = Query.parse(corpus, args.query, args.no_sentence_breaks)
            except ValueError as err:
                logging.info(f"Couldn't parse query {args.query}: {err}")
                continue
            logging.info(f"Query: {query}")

            results = search_corpus(query, args)
            logging.info(f"Results: {results}")

            stats: Counter[tuple[str, ...]] = Counter()
            query_offset = query.max_offset() + 1
            strings = corpus.tokens[gfeat]

            sampling_step = 1
            if args.sampling > 0:
                sampling_step = 1 + len(results) // args.sampling
            sampled_indices = range(0, len(results), sampling_step)
            if sampling_step > 1:
                logging.info(f"Too many results ({len(results)}): sampling {len(sampled_indices)} results")
            for i in sampled_indices:
                match_pos = results[i]
                group = tuple(
                    strings.interned_string(strings[p])
                    for p in range(match_pos, match_pos + query_offset)
                )
                stats[group] += 1

            sums = {
                "absolute": sum(stats.values()),
                "total-hits": len(results),
                "sampled-hits": len(sampled_indices),
                "corpus-size": len(corpus),
            }
            sums["relative"] = 1_000_000 * sums["absolute"] * sums["total-hits"]
            sums["relative"] /= sums["sampled-hits"] * sums["corpus-size"]  # type: ignore
            corpus_stats[corpus.name] = {"rows": [], "sums": sums}
            for group, hits in stats.items():
                corpus_stats[corpus.name]["rows"].append({
                    "value": {args.group_by: group},
                    "absolute": hits,
                    "relative": 1_000_000 * hits * sums["total-hits"] / sums["sampled-hits"] / sums["corpus-size"],
                })
                total_stats[group] += hits
            corpus_stats[corpus.name]["rows"].sort(key = lambda v: v["absolute"], reverse = True)  # type: ignore
            if args.num > 0:
                del corpus_stats[corpus.name]["rows"][args.num:]
            total_tokens += len(corpus)

    sums = {
        "absolute": sum(crp["sums"]["absolute"] for crp in corpus_stats.values()),
        "total-hits": sum(crp["sums"]["total-hits"] for crp in corpus_stats.values()),
        "sampled-hits": sum(crp["sums"]["sampled-hits"] for crp in corpus_stats.values()),
        "corpus-size": sum(crp["sums"]["corpus-size"] for crp in corpus_stats.values()),
    }
    sums["relative"] = 1_000_000 * sums["absolute"] * sums["total-hits"]
    sums["relative"] /= sums["sampled-hits"] * sums["corpus-size"]  # type: ignore
    combined_stats: dict[str, Any] = {"rows": [], "sums": sums}
    for group, hits in total_stats.items():
        combined_stats["rows"].append({
            "value": {args.group_by: group},
            "absolute": hits,
            "relative": 1_000_000 * hits * sums["total-hits"] / sums["sampled-hits"] / sums["corpus-size"],
        })
    count = len(total_stats)
    combined_stats["rows"].sort(key = lambda v: v["absolute"], reverse = True)  # type: ignore
    if args.num > 0:
        del combined_stats["rows"][args.num:]
    logging.info(f"Done, found ({count}) groups")

    return {
        "time": time.time() - start_time,
        "count": count,
        "corpora": corpus_stats,
        "combined": combined_stats,
    }


################################################################################
## Main cmd-line function: print a nice table

def main(args: argparse.Namespace) -> None:
    result = main_count(args)
    corpora = [("∑", result["combined"])]
    corpora += result["corpora"].items()
    table: dict[str, list[float]] = {}
    for n, (_, stat) in enumerate(corpora):
        table.setdefault("∑", []).append(stat["sums"][args.show])
        for row in stat["rows"]:
            hdr = " ".join(row["value"][args.group_by])
            table.setdefault(hdr, []).append(row[args.show])
        for row in table.values():
            while len(row) <= n:
                row.append(0)

    row_hdrs = sorted(table, key = lambda h: table[h][0], reverse = True)
    del row_hdrs[args.num:]
    hdr_width = 1 + max(len(hdr) for hdr in row_hdrs)
    col_width = max(10, 2 + max(len(crp) for crp,_ in corpora))
    print(f"{'Corpus':{hdr_width}} | ", end="")
    print(" |".join(f"{crp:>{col_width}}" for crp,_ in corpora))
    print(f"{'Sampling':{hdr_width}} | ", end="")
    print(" |".join(f"{stat['sums']['sampled-hits']/stat['sums']['total-hits']:>{col_width}.1%}" for _,stat in corpora))
    print("-"*hdr_width + "-+-" + "-+".join("-"*col_width for _ in corpora) + "--")
    precision = 0 if args.show == "absolute" else 1
    for hdr in row_hdrs:
        print(f"{hdr:<{hdr_width}} | " + " |".join(f"{n:>{col_width}.{precision}f}" for n in table[hdr]))


################################################################################
## Command-line arguments

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', nargs='+', required=True,
    help='name(s) of compiled corpora to search in')
parser.add_argument('--query', '-q', type=str, required=True,
    help='the query (e.g., \'[pos="ART"] [lemma="small"] [pos="SUBST"]\')')

parser.add_argument('--base-dir', '-d', type=Path, metavar='DIR', default=Path('corpora'),
    help='directory where to find the corpus (default: ./corpora/)')
parser.add_argument('--cache-dir', type=Path, metavar='DIR', default=Path('cache'),
    help='directory where to store cache files (default: ./cache/)')

parser.add_argument('--group-by', '-g', type=str, default="word",
    help='feature to group by (default: "word")')
parser.add_argument('--absolute', '-a', action="store_const", dest="show", const="absolute", default="relative",
    help='show absolute counts (default: show relative)')
parser.add_argument('--num', '-n', type=int, default=10,
    help='n:o of shown results (default: 10)')
parser.add_argument('--sampling', '--max', type=int, default=100_000,
    help='max n:o results to sample statistics from (default: 100_000)')

parser.add_argument('--no-cache', action="store_true", help="don't use cached queries")
parser.add_argument('--no-diskarray', action="store_true", help="don't use on-disk arrays")
parser.add_argument('--no-binary', action="store_true", help="don't use binary indexes")
parser.add_argument('--internal-merge', action='store_true',
    help='use the internal (slow) merge, even if the external Cython "fast-merge" is compiled')
parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO,
    help='verbose output')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING,
    help='debugging output')

parser.add_argument('--no-sentence-breaks', action='store_true',
    help="don't care about sentence breaks (default: do care)")
parser.add_argument('--filter', action='store_true',
    help='filter the final results (should not be necessary, and can take time)')

if __name__ == '__main__':
    args = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    main(args)
