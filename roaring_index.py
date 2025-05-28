
import sys
from tqdm import tqdm
from pathlib import Path

from pyroaring import BitMap

from disk import SymbolArray, SymbolCollection
from corpus import Corpus
from index import Template, TemplateLiteral, UnaryIndex, mkInstance
from indexset import IndexSet
from util import Feature, binsearch_first, binsearch_last


def generate_bitmaps_simple(index: UnaryIndex, fstrings: SymbolCollection):
    for i in tqdm(range(1, len(fstrings))):
        feat = fstrings.to_symbol(fstrings.to_name(i))
        instance = mkInstance([feat])
        start, end = index.lookup_instance(instance)
        iset = IndexSet(index.index, path=index.path, start=start, size=end-start+1)
        yield BitMap(iset).serialize()


def generate_bitmaps_fast(index: UnaryIndex, fstrings: SymbolCollection):
    tmpl = index.template.template[0]
    features = index.corpus.tokens[tmpl.feature]
    index_array = index.index.array
    def search_key(k: int):
        return features[index_array[k]]
    end = -1
    for i in tqdm(range(1, len(fstrings))):
        feat = fstrings.to_symbol(fstrings.to_name(i))
        if end < 0:
            start = binsearch_first(0, len(index)-1, feat, search_key)
        else:
            start = end + 1
        end = binsearch_last(start, len(index)-1, feat, search_key)
        iset = IndexSet(index.index, path=index.path, start=start, size=end-start+1)
        yield BitMap(iset).serialize()


def convert_unary(corpus: Corpus, feature: Feature, dbfile: Path):
    tmpl = TemplateLiteral(0, feature)
    index = UnaryIndex(corpus, Template([tmpl]))
    assert tmpl.offset == 0
    fstrings = corpus.symbols(feature)
    generate_bitmaps = generate_bitmaps_fast
    generate_bitmaps = generate_bitmaps_simple
    SymbolArray.create(dbfile, generate_bitmaps(index, fstrings), len(fstrings))



def build_bitmaps(corpus: Corpus, feature: Feature, dbfile: Path):
    fstrings = corpus.symbols(feature)
    tokens = corpus.tokens[feature]
    bitmaps: dict[int, BitMap] = {} #[BitMap() for _ in range(len(fstrings))]
    for pos, feat in tqdm(enumerate(tokens), total=len(tokens)):
        bitmaps.setdefault(feat, BitMap()).add(pos)
    for i in range(1, len(fstrings)-1, len(fstrings)//9):
        print(f"{fstrings.to_name(i).decode()} ({i}): {len(bitmaps[i])} : {bitmaps[i].min()}..{bitmaps[i].max()} = {bitmaps[i].serialize()[:5]}..{bitmaps[i].serialize()[-5:]}")
    def serialize():
        for feat in sorted(bitmaps):
            yield bitmaps[feat].serialize()
    SymbolArray.create(dbfile, serialize(), len(bitmaps))


def main(corpusfile: str, featstr: str, dbfile: str):
    corpus = Corpus(corpusfile)
    feature = Feature(featstr.encode())
    buildBMs = convert_unary
    buildBMs = build_bitmaps
    buildBMs(corpus, feature, Path(dbfile))



if __name__ == '__main__':
    main(*sys.argv[1:])

