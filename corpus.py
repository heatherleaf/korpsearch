
import re
import json
from pathlib import Path
import logging
from typing import Any
from contextlib import ExitStack
from collections.abc import Iterator, Sequence

from corpus_reader import corpus_reader
from disk import DiskIntArray, DiskStringArray, InternedString, InternedRange, StringCollection
from util import progress_bar, add_suffix, binsearch_last, Feature, FValue, SENTENCE

################################################################################
## Corpus

class Corpus:
    dir_suffix = '.corpus'
    features_file = 'features.cfg'
    feature_prefix = 'feature:'
    sentences_path = 'sentences'

    features: list[Feature]
    tokens: dict[Feature, DiskStringArray]
    sentence_pointers: DiskIntArray
    path: Path

    def __init__(self, corpus: Path) -> None:
        self.path = Path(corpus)
        if self.path.suffix != self.dir_suffix:
            self.path = add_suffix(self.path, self.dir_suffix)
        with open(self.path / self.features_file, 'r') as IN:
            self.features = [feat.encode() for feat in json.load(IN)]
        self.sentence_pointers = DiskIntArray(self.path / self.sentences_path)
        self.tokens = {
            feature: DiskStringArray(self.indexpath(self.path, feature))
            for feature in self.features
        }
        assert all(
            len(self) == len(arr) for arr in self.tokens.values()
        )
        
    def __str__(self) -> str:
        return f"[Corpus: {self.path.stem}]"

    def __repr__(self) -> str:
        return f"Corpus({self.path})"

    def __len__(self) -> int:
        return len(self.tokens[self.features[0]])

    def strings(self, feature: Feature) -> StringCollection:
        return self.tokens[feature]._strings  # type: ignore

    def intern(self, feature: Feature, value: FValue) -> InternedString:
        return self.tokens[feature].intern(value)

    def interned_range(self, feature: Feature, value: FValue) -> InternedRange:
        return self.tokens[feature].interned_range(value)

    def lookup_value(self, feature: Feature, i: InternedString) -> FValue:
        return FValue(self.tokens[feature].interned_bytes(i))

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
            features = self.features[:1]
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
                strings.interned_string(strings[p])
                for feat in features 
                for strings in [self.tokens[feat]]
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
        for sa in self.tokens.values(): sa.close()
        DiskIntArray.close(self.sentence_pointers)


    def sanity_check(self) -> None:
        logging.info("Sanity checking corpus")
        for feat in self.features:
            logging.debug(f"Checking corpus index: {feat.decode()}")
            self.tokens[feat].sanity_check()
        assert self.sentence_pointers.path
        logging.debug(f"Checking sentence pointers: {self.sentence_pointers.path.name}")
        sent_index = self.tokens[SENTENCE]
        sent_pointers = self.sentence_pointers.array
        assert sent_pointers[0] == sent_pointers[1] == 0, "Sentence 0 should not exist"
        sentence = 1
        sval = sent_index.intern(SENTENCE)
        for token, sval_ in enumerate(progress_bar(sent_index, "Checking sentences")):
            if sval == sval_:
                assert sent_pointers[sentence] == token, f"Sentence {sentence} doesn't point to the right token position"
                sentence += 1
        assert sentence == len(sent_pointers)
        logging.info("Done checking corpus")


    def get_matches(self, feature: Feature, match_regex: str) -> list[InternedString]:
        contains = False
        string_collection = self.strings(feature)
        strings = string_collection.strings
        positions = string_collection.starts.array
        # if match_regex.startswith(".*") and match_regex.endswith(".*"):
        #     match_regex = match_regex.strip(".*")
        #     contains = True
        binary_match_regex = bytes(match_regex, "utf-8")
        matches = (value.span() for value in re.finditer(binary_match_regex, strings))
        real_matches: list[InternedString] = []
        # Can be optimized
        for match in matches:
            start, end = 0, len(positions)-2
            while start <= end:
                mid = (start + end) // 2
                key = positions[mid]
                if key < match[0]:
                    start = mid + 1
                elif key > match[0]:
                    end = mid - 1
                else:
                    start = mid
                    break
            start_of_word = min(start, end)
            start_of_this_word = positions[start_of_word]
            start_of_next_word = positions[start_of_word+1]
            # Range is until
            # if contains and (match[1] - 1) < start_of_next_word:
            #     real_matches.append(InternedString(string_collection, start_of_word))
            if not contains and match[0] == start_of_this_word and match[1] == start_of_next_word-1:
                real_matches.append(InternedString(start_of_word))
        return real_matches


    @staticmethod
    def indexpath(basepath: Path, feature: Feature) -> Path:
        return basepath / (Corpus.feature_prefix + feature.decode()) / feature.decode()


    @staticmethod
    def build(basedir: Path, corpusfile: Path) -> None:
        logging.debug(f"Building corpus index from file: {str(corpusfile)}")

        with corpus_reader(corpusfile, "Collecting strings") as corpus:
            with open(basedir / Corpus.features_file, 'w') as OUT:
                json.dump([feat.decode() for feat in corpus.header], OUT)

            features = corpus.header
            stringsets: list[set[FValue]] = [set() for _feature in corpus.header]
            n_sentences = n_tokens = 0
            for sentence in corpus.sentences():
                n_sentences += 1
                for token in sentence:
                    n_tokens += 1
                    for strings, value in zip(stringsets, token):
                        strings.add(value)
        logging.debug(f" --> read {sum(map(len, stringsets))} distinct strings, {n_sentences} sentences, {n_tokens} tokens")

        path = basedir / Corpus.sentences_path
        with DiskIntArray.create(n_sentences+1, path, max_value = n_tokens) as sentence_array:
            sentence_array[0] = 0  # sentence 0 doesn't exist

            with ExitStack() as stack:  # to close all feature builders at once
                feature_builders: list[DiskStringArray] = []
                for feature, strings in zip(features, stringsets):
                    path = Corpus.indexpath(basedir, feature)
                    path.parent.mkdir(exist_ok=True)
                    str_array = stack.enter_context(DiskStringArray.create(path, strings, n_tokens))
                    feature_builders.append(str_array)

                corpus = stack.enter_context(corpus_reader(corpusfile, "Building indexes"))
                ctr = 0
                for n, sentence in enumerate(corpus.sentences(), 1):
                    sentence_array[n] = ctr
                    for token in sentence:
                        for builder, value in zip(feature_builders, token):
                            builder[ctr] = builder.intern(value)
                        ctr += 1
                assert ctr == n_tokens

        logging.info(f"Built corpus index, {n_tokens} tokens, {n_sentences} sentences")
