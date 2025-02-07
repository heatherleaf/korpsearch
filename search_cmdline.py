
import argparse
from pathlib import Path
import logging
import json

from util import setup_logger
from search import main_search


def main(args: argparse.Namespace) -> None:
    out = main_search(args)
    if args.print == 'json':
        print(json.dumps(out))
    elif args.print == 'kwic' and out['kwic']:
        print(f"{out['hits']} search results. Showing n:o {out['start']}-{out['end']}:")
        for n, result in enumerate(out['kwic'], out['start']):
            match = result['match']
            print(f"{n}. [{result['sentence']}:{match['position']}]", end="")
            for i, token in enumerate(result['tokens']):
                if i == match['start']:
                    print(" {", end="")
                print(" " + "/".join(token.values()), end="")
                if i + 1 == match['end']:
                    print(" }", end="")
            print()
    else:
        print(f"{out['hits']} search results.")


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', type=Path, required=True,
    help='path to compiled corpus')
parser.add_argument('--query', '-q', type=str, required=True,
    help='the query (e.g., \'[pos="ART"] [lemma="small"] [pos="SUBST"]\')')

parser.add_argument('--print', '-p', choices=['kwic','json'], default='kwic',
    help='output format for search results (default: kwic = keywords in context)')
parser.add_argument('--start', '-s', type=int, default=0,
    help='index of first result (default: 0)')
parser.add_argument('--num', '-n', type=int, default=10,
    help='n:o of shown results (default: 10)')
parser.add_argument('--end', '-e', type=int,
    help='index of last result (default: decided by --start and --num)')
parser.add_argument('--show', '-f', type=str,
    help='comma-separated list of features to show (default: the ones in the query)')

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
