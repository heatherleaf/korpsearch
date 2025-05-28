
from pathlib import Path
from argparse import Namespace
import logging

from pyroaring import BitMap

from disk import SymbolArray
from corpus import Corpus
from index import Template, TemplateLiteral, Index, UnaryIndex
from util import progress_bar, binsearch_first


def build_index(corpus: Corpus, template: Template, args: Namespace) -> None:
    index_path = Index.indexpath(corpus, template)
    index_path.parent.mkdir(exist_ok=True)
    if len(template) == 1:
        build_unary_index(corpus, index_path, template, args)
    elif len(template) == 2:
        build_binary_index(corpus, index_path, template, args)
    else:
        raise ValueError(f"Cannot build indexes of length {len(template)}: {template}")
    with SymbolArray(index_path) as suffix_array:
        logging.info(f"Built index for {template}, with {len(suffix_array)} elements")



def build_unary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building unary index: {template}")
    assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
    assert not template.literals, f"Cannot build UnaryIndex from templates with literals: {template}"

    tmpl = template.template[0]
    assert tmpl.offset == template.min_offset() == template.max_offset() == 0
    features = corpus.tokens[tmpl.feature].raw()

    bitmaps: dict[int, BitMap] = {}
    for pos, value in enumerate(progress_bar(features, desc="Collecting positions")):
        if value:
            bitmaps.setdefault(value, BitMap()).add(pos)

    logging.debug(f"Creating index file")
    with SymbolArray.create(index_path,
                                (bitmaps[value].serialize() for value in sorted(bitmaps)),
                                len(bitmaps),
                            ) as builder:



def build_binary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building binary index: {template}")
    assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
    assert all(lit.negative for lit in template.literals), f"Cannot handle positive template literals: {template}"

    tmpl1, tmpl2 = template.template
    assert tmpl1.offset == template.min_offset() == 0
    assert tmpl2.offset == template.max_offset()
    features1 = corpus.tokens[tmpl1.feature].raw()
    features2 = corpus.tokens[tmpl2.feature].raw()
    min_freq = args.min_frequency

    unary1 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl1.feature)]))
    unary2 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl2.feature)]))

    # Optimisation: cache index lookups, because the size of the vocabulary
    # is much smaller than the corpus size.
    cache1: dict[int, bool] = {}
    cache2: dict[int, bool] = {}
    def unary_min_frequency(unary: UnaryIndex, unary_key: int, cache: dict[int, bool]) -> bool:
        if unary_key not in cache:
            search_key = unary.search_key()
            start = binsearch_first(0, len(unary)-1, unary_key, search_key)
            end = start + min_freq - 1
            cache[unary_key] = (end < len(unary) and search_key(end) == unary_key)
        return cache[unary_key]

    # Optimisation: use the underlying memoryview for each DiskStringArray,
    # and don't bother with looking up InternedStrings - use the ints directly instead.
    test_literals = [(corpus.tokens[lit.feature].raw(), lit.offset, lit.value) for lit in template.literals]

    bitmaps: dict[int, BitMap] = {}
    skipped_instances = 0
    for pos in progress_bar(range(len(corpus) - template.max_offset()), desc="Collecting positions"):
        val1, val2 = features1[pos], features2[pos + tmpl2.offset]
        if val1 and val2 and all(lfeat[pos + loffset] != lval for (lfeat, loffset, lval) in test_literals):
            if min_freq and not (
                        unary_min_frequency(unary1, val1, cache1) and
                        unary_min_frequency(unary2, val2, cache2)
                    ):
                skipped_instances += 1
            else:
                value = (val1 << 32) + val2
                bitmaps.setdefault(value, BitMap()).add(pos)
    if skipped_instances:
        logging.debug(f"Skipped {skipped_instances} low-frequency instances")

    logging.debug(f"Creating index file")
    SymbolArray.create(
        index_path,
        (bitmaps[value].serialize() for value in sorted(bitmaps)),
        len(bitmaps),
    )

