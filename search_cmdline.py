
import argparse
from pathlib import Path
import logging
import json

from util import setup_logger
from search import main_search


def main(args:argparse.Namespace):
    out = main_search(args)
    if args.print == 'json':
        print(json.dumps(out))
    elif args.print == 'kwic':
        print(f"{out['total-matches']} search results:")
        for n, result in enumerate(out['matches'], out['first-match']):
            print(f"{n}. [{result['sentence']}:{result['pos']}]", end="")
            for i, token in enumerate(result['tokens']):
                if i == result['start']:
                    print(" {", end="")
                if i == result['start'] + result['length']:
                    print(" }", end="")
                print(" " + "/".join(token.values()), end="")
            print()


################################################################################
## Main

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', type=Path, required=True, help='path to compiled corpus')
parser.add_argument('--query', '-q', type=str, required=True, help='the query')

parser.add_argument('--print', '-p', choices=['kwic','json'], default='kwic', 
                    help='output format for search results (default: kwic = keywords in context)')
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
    main(args)
