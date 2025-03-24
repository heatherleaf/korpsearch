
import argparse
import logging
import os
from pathlib import Path

from disk import DiskIntArray
from util import setup_logger


def trim_cache(args: argparse.Namespace) -> None:
    mb = 1 << 20
    cache_files = sorted(args.cache_dir.rglob('*' + DiskIntArray.array_suffix),
                         key = lambda f: f.stat().st_mtime)
    size = sum(file.stat().st_size for file in cache_files)
    logging.info(f"Current cache size: {size/mb:.1f} MB (max limit: {args.max_size} MB)")
    excess = size - args.max_size * mb
    for file in cache_files:
        if excess <= 0:
            break
        logging.info(f"Deleting old cache file: {file} ({file.stat().st_size/mb:.1f} MB)")
        excess -= file.stat().st_size
        DiskIntArray.getconfig(file).unlink(missing_ok=True)
        file.unlink(missing_ok=True)
    for dir, _, filenames in os.walk(args.cache_dir, topdown=False):
        dir = Path(dir)
        if dir != args.cache_dir:
            if filenames == ["__info__"]:
                (dir / filenames[0]).unlink()
            try:
                dir.rmdir()
                logging.info(f"Removed empty cache directory: {dir}")
            except OSError:
                pass


################################################################################
## Command-line arguments

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--cache-dir', '-d', type=Path, metavar='DIR', default=Path('cache'),
    help='directory where to store cache files (default: ./cache/)')
parser.add_argument('--max-size', '-m', type=int, default=100,
    help='total max size of cache, in MB (default: 100 MB)')
parser.add_argument('--quiet', '-q', action="store_const", dest="loglevel", const=logging.WARNING, default=logging.INFO,
    help="only show warnings (default: be more verbose)")

if __name__ == '__main__':
    args = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    trim_cache(args)
