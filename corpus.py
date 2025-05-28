
import json
from pathlib import Path
import logging
from typing import Any
from argparse import Namespace
from contextlib import ExitStack
from collections.abc import Iterator, Sequence

from corpus_reader import corpus_reader
from disk import IntArray, SymbolArray, Symbol, SymbolRange, SymbolCollection
from util import progress_bar, add_suffix, binsearch_last, Feature, FValue, SENTENCE

################################################################################
## Corpus

class Corpus:
    dir_suffix = '.corpus'
    features_file = 'features.cfg'
    feature_prefix = 'feature:'
    sentences_path = 'sentences'

    tokens: dict[Feature, SymbolArray]
    sentence_pointers: IntArray
    id: str
    name: str
    path: Path
    base_dir: Path

    def __init__(self, corpus: str, base_dir: Path = Path()) -> None:
        self.name = corpus
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / corpus
        self.id = str(self.path)
        if self.path.suffix != self.dir_suffix:
            self.path = add_suffix(self.path, self.dir_suffix)
        self.sentence_pointers = IntArray(self.path / self.sentences_path)
        self.tokens = {
            feature: SymbolArray(self.indexpath(self.path, feature))
            for feature in self.features()
        }
        assert all(
            len(self) == len(arr) for arr in self.tokens.values()
        )

    def __str__(self) -> str:
        return f"[Corpus: {self.name}]"

    def __repr__(self) -> str:
        return f"Corpus({self.name}, base={self.base_dir})"

    def __len__(self) -> int:
        return len(self.tokens[self.features()[0]])

    def default_feature(self) -> Feature:
        return self.features()[0]

    def features(self) -> list[Feature]:
        with open(self.path / self.features_file, 'r') as IN:
            return [feat.encode() for feat in json.load(IN)]

    def symbols(self, feature: Feature) -> SymbolCollection:
        return self.tokens[feature].symbols  # type: ignore

    def get_symbol(self, feature: Feature, value: FValue) -> Symbol:
        return self.tokens[feature].symbols.to_symbol(value)

    def get_symbol_range(self, feature: Feature, prefix: FValue) -> SymbolRange:
        return self.tokens[feature].symbols.to_symbol_range(prefix)

    def lookup_symbol(self, feature: Feature, sym: Symbol) -> FValue:
        return FValue(self.tokens[feature].symbols.to_name(sym))

    def num_sentences(self) -> int:
        return len(self.sentence_pointers)-1

    def sentences(self) -> Iterator[range]:
        sents = self.sentence_pointers.array
        for start, end in zip(sents[1:], sents[2:]):
            yield range(start, end)
        yield range(sents[len(sents)-1], len(self))

    def sentence_positions(self, n: int) -> range:
        sents = self.sentence_pointers.array
        start = sents[n]
        nsents = len(sents)
        end = sents[n+1] if n+1 < nsents else len(self)
        return range(start, end)

    def render_sentence(self, sent: int, pos: int = -1, offset: int = -1,
                        features: Sequence[Feature] = (), context: int = -1) -> str:
        if not features:
            features = [self.default_feature()]
        tokens: list[str] = []
        positions = self.sentence_positions(sent)
        for p in positions:
            if p < 0 or p >= len(self):
                continue
            if context >= 0:
                if p < pos-context:
                    continue
                if p == pos-context and p > positions.start:
                    tokens.append('...')
            if p == pos:
                tokens.append('[')
            tokens.append('/'.join(
                symbol_array.symbols.to_name(symbol_array[p]).decode()
                for feat in features
                for symbol_array in [self.tokens[feat]]
            ))
            if p == pos+offset:
                tokens.append(']')
            if context >= 0:
                if p == pos+offset+context and p < positions.stop:
                    tokens.append('...')
                    break
        return ' '.join(tokens)

    def get_sentence_from_position(self, pos: int) -> int:
        ptrs = self.sentence_pointers.array
        return binsearch_last(0, len(ptrs)-1, pos, lambda k: ptrs[k], error=False)

    def __enter__(self) -> 'Corpus':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        for sa in self.tokens.values():
            sa.close()
        IntArray.close(self.sentence_pointers)


    def sanity_check(self) -> None:
        logging.info("Sanity checking corpus")
        for feat in self.features():
            logging.debug(f"Checking corpus index: {feat.decode()}")
            self.tokens[feat].sanity_check()
        assert self.sentence_pointers.path
        logging.debug(f"Checking sentence pointers: {self.sentence_pointers.path.name}")
        sentence_index = self.tokens[SENTENCE]
        sentence_pointers = self.sentence_pointers.array
        assert sentence_pointers[0] == sentence_pointers[1] == 0, "Sentence 0 should not exist"
        sentence = 1
        sval = sentence_index.symbols.to_symbol(SENTENCE)
        for token, sval_ in enumerate(progress_bar(sentence_index, "Checking sentences")):
            if sval == sval_:
                assert sentence_pointers[sentence] == token, f"Sentence {sentence} doesn't point to the right token position"
                sentence += 1
        assert sentence == len(sentence_pointers)
        logging.info("Done checking corpus")


    def get_matches(self, feature: Feature, regex: str) -> list[Symbol]:
        symbols = self.symbols(feature)
        return list(symbols.finditer(regex.encode()))


    @staticmethod
    def indexpath(basepath: Path, feature: Feature) -> Path:
        return basepath / (Corpus.feature_prefix + feature.decode()) / feature.decode()


    @staticmethod
    def build(basedir: Path, corpusfile: Path, args: Namespace = Namespace()) -> None:
        logging.debug(f"Building corpus index from file: {str(corpusfile)}")

        with corpus_reader(corpusfile, "Collecting strings", args) as corpus:
            with open(basedir / Corpus.features_file, 'w') as OUT:
                print(json.dumps([feat.decode() for feat in corpus.header()]), file=OUT)

            features = corpus.header()
            stringsets: list[set[FValue]] = [set() for _feature in corpus.header()]
            n_sentences = n_tokens = 0
            for sentence in corpus.sentences():
                n_sentences += 1
                for token in sentence:
                    n_tokens += 1
                    for strings, value in zip(stringsets, token):
                        strings.add(value)
        logging.debug(f" --> read {sum(map(len, stringsets))} distinct strings, {n_sentences} sentences, {n_tokens} tokens")

        path = basedir / Corpus.sentences_path
        with IntArray.create(n_sentences+1, path, max_value = n_tokens) as sentence_array:
            sentence_array[0] = 0  # sentence 0 doesn't exist

            with ExitStack() as stack:  # to close all feature builders at once
                feature_builders: list[SymbolArray] = []
                for feature, strings in zip(features, stringsets):
                    path = Corpus.indexpath(basedir, feature)
                    path.parent.mkdir(exist_ok=True)
                    str_array = stack.enter_context(SymbolArray.create(path, strings, n_tokens))
                    feature_builders.append(str_array)

                corpus = stack.enter_context(corpus_reader(corpusfile, "Building indexes", args))
                ctr = 0
                for n, sentence in enumerate(corpus.sentences(), 1):
                    sentence_array[n] = ctr
                    for token in sentence:
                        for builder, value in zip(feature_builders, token):
                            builder[ctr] = builder.symbols.to_symbol(value)
                        ctr += 1
                assert ctr == n_tokens

        logging.info(f"Built corpus index, {n_tokens} tokens, {n_sentences} sentences")
