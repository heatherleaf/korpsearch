
import argparse
import logging
import os
from pathlib import Path

from util import file_size


def trim_cache(args: argparse.Namespace) -> None:
    cache_dir: Path = args.cache_dir
    mb = 1 << 20
    cache_files = sorted(cache_dir.rglob('*.bitmap'), key = lambda f: f.stat().st_mtime)
    file_sizes = [file_size(file) for file in cache_files]
    total_size = sum(file_sizes)
    logging.info(f"Current cache size: {total_size/mb:.1f} MB (max limit: {args.max_size} MB)")
    excess = total_size - args.max_size * mb
    for file, size in zip(cache_files, file_sizes):
        if excess <= 0:
            break
        logging.info(f"Deleting old cache file: {file} ({size/mb:.1f} MB)")
        excess -= size
        file.unlink(missing_ok=True)
    for dir, _, filenames in os.walk(cache_dir, topdown=False):
        dir = Path(dir)
        if dir != cache_dir:
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
parser.add_argument('--quiet', '-q', action="store_const", dest="log_level", const=logging.WARNING, default=logging.INFO,
    help="only show warnings (default: be more verbose)")
parser.add_argument('--log-file', '-f', type=Path,
    help="log file (default: print to stderr)")

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(
        format = "%(asctime)s | %(message)s",
        datefmt = "%Y-%m-%d, %H:%M:%S",
        level = args.log_level,
        filename = args.log_file,
    )
    trim_cache(args)
