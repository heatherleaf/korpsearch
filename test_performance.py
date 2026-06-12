
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

from search import main_search

def test_all_queries(args: Namespace) -> None:
    args.no_cache = True
    args.num = args.start = args.end = 0
    args.show = args.no_sentence_breaks = args.filter = args.loglevel = None
    queries: list[str] = []
    for query_file in args.queries:
        with open(query_file) as file:
            queries += [query for line in file for query in [line.strip()] if query and not query.startswith("#")]
    elapsed = 0
    for n, query in enumerate(queries, 1):
        args.query = query
        total = elapsed * (len(queries) - n) / n
        sys.stderr.write(f"Testing {n} (of {len(queries)})... elapsed {elapsed:.0f} s (of {total:.0f} s): {query}\r")
        try:
            result = main_search(args)
            print(json.dumps({
                'n': n,
                'query': query,
                'hits': result['hits'],
                'time': result['time'],
            }))
            elapsed += result['time']
        except Exception as err:
            print(json.dumps({
                'n': n,
                'query': query,
                'error': repr(err),
            }))
        sys.stderr.write("\033[K") # Clear the line
    print(f"Done testing {len(queries)} queries, total time: {elapsed:.0f} s", file=sys.stderr)


parser = ArgumentParser(description='Run performance test on a given corpus')

parser.add_argument('queries', type=Path, nargs='+',
    help='files that contain queries, one per line')
parser.add_argument('--corpus', '-c', nargs=1, required=True,
    help='name of compiled corpus to search in')
parser.add_argument('--base-dir', '-d', type=Path, metavar='DIR', default=Path('corpora'),
    help='directory where to find the corpus (default: ./corpora/)')
parser.add_argument('--no-binary', action="store_true",
    help="don't use binary indexes")


if __name__ == '__main__':
    test_all_queries(parser.parse_args())
