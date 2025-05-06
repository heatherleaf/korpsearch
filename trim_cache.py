
import argparse
import logging
import os
from pathlib import Path

from disk import DiskIntArray


def trim_cache(args: argparse.Namespace) -> None:
    mb = 1 << 20
    cache_files = sorted(args.cache_dir.rglob('*' + DiskIntArray.array_suffix),
                         key = lambda f: f.stat().st_mtime)
    size = sum(DiskIntArray.disksize(file) for file in cache_files)
    logging.info(f"Current cache size: {size/mb:.1f} MB (max limit: {args.max_size} MB)")
    excess = size - args.max_size * mb
    for file in cache_files:
        if excess <= 0:
            break
        logging.info(f"Deleting old cache file: {file} ({file.stat().st_size/mb:.1f} MB)")
        excess -= DiskIntArray.disksize(file)
        DiskIntArray.getconfig(file).unlink(missing_ok=True)
        DiskIntArray.getpath(file).unlink(missing_ok=True)
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
