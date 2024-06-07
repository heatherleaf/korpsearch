
import json
from pathlib import Path
import logging
from typing import Any
from contextlib import ExitStack
from collections.abc import Iterator, Sequence

from disk import DiskIntArray, DiskStringArray, InternedString
from util import progress_bar, ProgressBar, CompressedFileReader, add_suffix, binsearch_last

################################################################################
## Corpus


class Corpus:
    dir_suffix = '.corpus'
    features_file = 'features.cfg'
    feature_prefix = 'feature:'
    sentences_path = 'sentences'
    sentence_feature = 's'
    sentence_start_value = b'S'
    empty_value = b''

    features: list[str]
    tokens: dict[str, DiskStringArray]
    sentence_pointers: DiskIntArray
    path: Path

    def __init__(self, corpus: Path) -> None:
        self.path = Path(corpus)
        if self.path.suffix != self.dir_suffix:
            self.path = add_suffix(self.path, self.dir_suffix)
        with open(self.path / self.features_file, 'r') as IN:
            self.features = json.load(IN)
        self.sentence_pointers = DiskIntArray(self.path / self.sentences_path)
        self.tokens = {
            feature: DiskStringArray(self.path / (self.feature_prefix + feature) / feature)
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

    def intern(self, feature: str, value: bytes) -> InternedString:
        return self.tokens[feature].intern(value)

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
        end = sents[n+1] if n+1 < nsents else nsents
        return range(start, end)

    def render_sentence(self, sent: int, pos: int = -1, offset: int = -1, 
                        features: Sequence[str] = (), context: int = -1) -> str:
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
            tokens.append('/'.join(str(self.tokens[feat][p]) for feat in features))
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

    @staticmethod
    def build_from_csv(basedir: Path, csv_corpusfile: Path) -> None:
        logging.debug(f"Building corpus index")
        corpus = CompressedFileReader(csv_corpusfile)
        csv_filesize = corpus.file_size()

        # the first line in the CSV should be a header with the names of each column (=features)
        corpus.reader.seek(0)
        features: list[str] = corpus.reader.readline().decode().split()
        assert Corpus.sentence_feature not in features
        features.insert(0, Corpus.sentence_feature)

        with open(basedir / Corpus.features_file, 'w') as OUT:
            json.dump(features, OUT)

        def iterate_sentences(description: str) -> Iterator[list[list[bytes]]]:
            # Skip over the header line
            corpus.reader.seek(0)
            corpus.reader.readline()
            pbar: ProgressBar[None]
            with progress_bar(total=csv_filesize, desc=description) as pbar:
                sentence: list[list[bytes]] = []
                for line in corpus.reader:
                    pbar.update(corpus.file_position() - pbar.n)
                    line = line.strip()
                    if line.startswith(b'# '):
                        if sentence: 
                            yield sentence
                            sentence = []
                    elif line:
                        token: list[bytes] = line.split(b'\t')
                        token.insert(0, Corpus.empty_value if sentence else Corpus.sentence_start_value)
                        if len(token) < len(features):
                            token += [Corpus.empty_value] * (len(features) - len(token))
                        sentence.append(token)
                if sentence: 
                    yield sentence

        strings: list[set[bytes]] = [set() for _feature in features]
        n_sentences = n_tokens = 0
        for sentence in iterate_sentences("Collecting strings"):
            n_sentences += 1
            for token in sentence:
                n_tokens += 1
                for i, feat in enumerate(token):
                    strings[i].add(feat)
        logging.debug(f" --> read {sum(map(len, strings))} distinct strings, {n_sentences} sentences, {n_tokens} tokens")

        path = basedir / Corpus.sentences_path
        with DiskIntArray.create(n_sentences+1, path, max_value = n_tokens) as sentence_builder:
            sentence_builder[0] = 0  # sentence 0 doesn't exist

            with ExitStack() as stack:
                feature_builders: list[DiskStringArray] = []
                for i, feature in enumerate(features):
                    path = basedir / (Corpus.feature_prefix + feature) / feature
                    path.parent.mkdir(exist_ok=True)
                    feature_builders.append(stack.enter_context(
                        DiskStringArray.create(path, strings[i], n_tokens)
                    ))

                ctr = 0
                for n, sentence in enumerate(iterate_sentences("Building indexes"), 1):
                    sentence_builder[n] = ctr
                    for token in sentence:
                        for i, feat in enumerate(token):
                            feature_builders[i][ctr] = feat
                        ctr += 1
                assert ctr == n_tokens

        logging.info(f"Built corpus index, {n_tokens} tokens, {n_sentences} sentences")
