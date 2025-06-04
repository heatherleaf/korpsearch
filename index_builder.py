
from pathlib import Path
from argparse import Namespace
from collections.abc import Iterator
from mmap import mmap
import json
import logging

from pyroaring import BitMap

from disk import IntArray, Symbol, BytesArray, IntBytesMap
from corpus import Corpus
from literals import Template, TemplateLiteral, Instance
from index import Index, UnaryIndex
import sort
from util import add_suffix, progress_bar

# Possible sorting alternatives, the first is the default:
SORTER_CHOICES = ['bitmap', 'tmpfile', 'cython']


def build_index(corpus: Corpus, template: Template, args: Namespace) -> None:
    index_path = Index.indexpath(corpus, template)
    index_path.parent.mkdir(exist_ok=True)
    arity = len(template)
    logging.info(f"Building index {template}, using sorter {args.sorter}")
    if arity == 1:
        collect = collect_from_unary_template(corpus, template, args)
    elif arity == 2:
        collect = collect_from_binary_template(corpus, template, args)
    else:
        raise ValueError(f"Cannot build indexes of length {arity}: {template}")

    if args.sorter == 'bitmap':
        size = build_index_via_bitmaps(index_path, collect, arity, args)
    else:
        size = build_index_via_tmpfile(index_path, collect, arity, args)

    config: dict[str, int] = {}
    config['arity'] = arity
    config['size'] = size
    if arity > 1 and args.min_frequency:
        config['min_frequency'] = args.min_frequency
    with open(Index.getconfigpath(index_path), "w") as configfile:
        print(json.dumps(config), file=configfile)
    with Index(corpus, template) as index:
        smallsize = len(index.smallsets or ())
        nbigsets = len(index.bigsets or ())
        avgbigsize = round((len(index) - smallsize) / nbigsets) if nbigsets > 0 else 0
        logging.info(f"Built {template}, size {len(index):,d}: total {smallsize:,d} in small sets, the rest in {nbigsets:,d} big sets, avg. size {avgbigsize:,d}")


def collect_from_unary_template(corpus: Corpus, template: Template, args: Namespace) -> Iterator[tuple[int, int]]:
    assert not template.literals, f"Cannot build UnaryIndex from templates with literals: {template}"
    (tmpl,) = template.template
    assert tmpl.offset == template.min_offset() == template.max_offset() == 0
    features = progress_bar(corpus.tokens[tmpl.feature].raw(), desc="Collecting positions")
    return ((value, pos) for (pos, value) in enumerate(features) if value)


def collect_from_binary_template(corpus: Corpus, template: Template, args: Namespace) -> Iterator[tuple[int, int]]:
    assert all(lit.negative for lit in template.literals), f"Cannot handle positive template literals: {template}"
    (tmpl1, tmpl2) = template.template
    offset2 = tmpl2.offset
    assert tmpl1.offset == template.min_offset() == 0
    assert tmpl2.offset == template.max_offset() == offset2
    features1 = corpus.tokens[tmpl1.feature].raw()
    features2 = corpus.tokens[tmpl2.feature].raw()

    # Optimisation: use the underlying memoryview for each SymbolArray,
    # and don't bother with looking up symbols - use the ints directly instead.
    test_literals = [(corpus.tokens[lit.feature].raw(), lit.offset, lit.value) for lit in template.literals]

    size = len(corpus) - offset2
    min_freq = args.min_frequency
    if not min_freq:
        # The simple case: --min-frequency is not set, so don't filter our low-frequent features.
        return (
            ((val1 << 32) + val2, pos)
            for pos in progress_bar(range(size), desc="Collecting positions")
            for (val1, val2) in [(features1[pos], features2[pos + offset2])]
            if val1 and val2 and all(lfeat[pos + loffset] != lval for (lfeat, loffset, lval) in test_literals)
        )

    # The more complicated case: only record a binary instance if both features are common enough.
    # If --min-frequency is set to M, then both features have to have frequency at least M.

    # Already-compiled unary indexes which we use to look up the respective individual frequencies.
    unary1 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl1.feature)]))
    unary2 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl2.feature)]))

    # If `unary_min_frequency` returns True, then the feature occurs at least `min_freq` times.
    # Optimisation: Cache lookups, because the vocabulary size is much smaller than the corpus size.
    cache1: dict[int, bool] = {}
    cache2: dict[int, bool] = {}
    def unary_min_frequency(unary: UnaryIndex, unary_key: Symbol, cache: dict[int, bool]) -> bool:
        if unary_key not in cache:
            freq = len(unary.search(Instance([unary_key])))
            cache[unary_key] = freq >= min_freq
        return cache[unary_key]

    def collect() -> Iterator[tuple[int, int]]:
        skipped_instances = 0
        for pos in progress_bar(range(size), desc="Collecting positions"):
            val1, val2 = features1[pos], features2[pos + offset2]
            if val1 and val2 and all(lfeat[pos + loffset] != lval for (lfeat, loffset, lval) in test_literals):
                if unary_min_frequency(unary1, val1, cache1) and unary_min_frequency(unary2, val2, cache2):
                    yield ((val1 << 32) + val2, pos)
                else:
                    skipped_instances += 1
        if skipped_instances:
            logging.debug(f"Skipped {skipped_instances:,d} low-frequency instances")
    return collect()


def build_index_via_bitmaps(path: Path, collect: Iterator[tuple[int, int]], arity: int, args: Namespace) -> int:
    bitmaps: dict[int, BitMap] = {}
    size = 0
    for value, pos in collect:
        bitmaps.setdefault(value, BitMap()).add(pos)
        size += 1

    logging.debug(f"Sorting {len(bitmaps):,d} bitmaps, {size:,d} values")
    sorted_bitmaps = sorted(bitmaps.items(), key = lambda b:b[0])
    max_value = sorted_bitmaps[-1][0]

    logging.debug(f"Building index")
    smallsets = (n for (_, bitmap) in sorted_bitmaps if len(bitmap) < args.bigset_limit for n in bitmap)
    bigset_keys = (key for (key, bitmap) in sorted_bitmaps if len(bitmap) >= args.bigset_limit)
    bigset_values = (bitmap.serialize() for (_, bitmap) in sorted_bitmaps if len(bitmap) >= args.bigset_limit)
    IntArray.build(path, smallsets, size=size, max_value=size)
    IntBytesMap.build(path, bigset_keys, bigset_values, size=size, max_value=max_value)
    return size


def build_index_via_tmpfile(path: Path, collect: Iterator[tuple[int, int]], arity: int, args: Namespace) -> int:
    # First we collect all (value, position) pairs into a temporary binary file.
    tmppath = add_suffix(path, '.tmp')
    rowsize = (arity + 1) * 4
    size = max_value = 0
    with open(tmppath, 'w+b') as file:
        for value, pos in collect:
            row = (value << 32) + pos
            file.write(row.to_bytes(rowsize, 'big'))
            size += 1
            max_value = max(max_value, value)

    # Then we sort the temporary file, either using qsort from C (preferred) or a Python sort function.
    if args.sorter == 'cython':
        from faster_index_builder import sort_index  # type: ignore
        logging.debug(f"Sorting {size:,d} rows using C 'qsort'")
        sort_index(tmppath, size, rowsize)
    else:
        logging.debug(f"Sorting {size:,d} rows")
        with open(tmppath, 'r+b') as file:
            with mmap(file.fileno(), 0) as bytes_mmap:
                sort.quicksort(bytes_mmap, rowsize, cutoff = args.cutoff or 1_000_000)

    # Here is an iterator read the sorted file and yields each value together
    # with the set of positions for that value.
    def yield_bitmaps() -> Iterator[tuple[int, BitMap]]:
        with open(tmppath, 'rb') as file:
            prev_value = b''
            bitmap = BitMap()
            for _ in progress_bar(range(size), "Building index"):
                row = file.read(rowsize)
                value = row[:-4]
                if value != prev_value:
                    if bitmap:
                        yield (int.from_bytes(prev_value, 'big'), bitmap)
                        bitmap = BitMap()
                    prev_value = value
                bitmap.add(int.from_bytes(row[-4:], 'big'))
            if bitmap:
                yield (int.from_bytes(prev_value, 'big'), bitmap)

    # We use the iterator above to build two search indexes from the position sets above:
    #  - an IntArray for storing small sets
    #  - a IntBytesMap for storing large sets
    logging.debug(f"Building index")
    keyspath, valspath = IntBytesMap.getpaths(path)
    def yield_bigset_values() -> Iterator[bytes]:
        smallsets = IntArray.create(size, path, max_value=size)
        bigset_keys = IntArray.create(size, keyspath, max_value=max_value)
        smallsets_ctr = bigsets_ctr = 0
        for value, bitmap in yield_bitmaps():
            if len(bitmap) < args.bigset_limit:
                for n in bitmap:
                    smallsets[smallsets_ctr] = n
                    smallsets_ctr += 1
            else:
                bigset_keys[bigsets_ctr] = value
                bigsets_ctr += 1
                yield bitmap.serialize()
        smallsets.truncate(smallsets_ctr)
        smallsets.close()
        bigset_keys.truncate(bigsets_ctr)
        bigset_keys.close()
    BytesArray.build(valspath, yield_bigset_values())

    if not args.keep_tmpfiles:
        tmppath.unlink()
    return size
