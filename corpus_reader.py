import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Callable
from pathlib import Path
from typing import Iterable

from util import CompressedFileReader, uncompressed_suffix
from util import Feature, FValue, EMPTY, SENTENCE, START, check_feature
from util import progress_bar

Header = list[Feature]
Token = list[FValue]
Sentence = list[Token]


class CorpusReader(ABC):

    @property
    @abstractmethod
    def header(self) -> Header:
        pass

    @abstractmethod
    def sentences(self) -> Iterator[Sentence]:
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()


def corpus_reader(path: Path, description: str, sentence_feature: bool = True,
                  reversed_features: bool = True) -> CorpusReader:
    suffix = uncompressed_suffix(path)
    try:
        reader = CORPUS_READERS[suffix]
    except KeyError:
        raise ValueError(f"Cannot find a corpus reader for file type: {suffix}")
    reader_instance = reader(path, description)
    return AugmentedReader(reader_instance, sentence_feature, reversed_features)


class AugmentedReader(CorpusReader):
    wrapped: CorpusReader
    sentence_feature: bool
    reversed_features: bool

    def __init__(self, reader: CorpusReader, sentence_feature: bool, reversed_features: bool):
        self.wrapped = reader
        self.sentence_feature = sentence_feature
        self.reversed_features = reversed_features

        self._header = reader.header
        if self.reversed_features:
            revd = [Feature(feature + b'_rev') for feature in self._header]
            self._header.extend(revd)
        if self.sentence_feature:
            self._header.append(SENTENCE)

    @property
    def header(self) -> Header:
        return self._header

    def sentences(self) -> Iterator[Sentence]:
        for sentence in self.wrapped.sentences():
            for token in sentence:
                if self.reversed_features:
                    token += [FValue(val.decode()[::-1].encode()) for val in token]
                if self.sentence_feature:
                    token.append(EMPTY)
            if self.sentence_feature:
                sentence[0][-1] = START
            yield sentence

    def close(self):
        self.wrapped.close()


class CSVReader(CorpusReader):
    _corpus: CompressedFileReader
    _header: Header
    _description: str
    _n_feats: int

    def __init__(self, path: Path, description: str):
        self._description = description
        self._corpus = CompressedFileReader(path)

        header_line = self._corpus.reader.readline()
        self._header = [Feature(f) for f in CSVReader.split_line(header_line.strip())]
        for feat in self._header: check_feature(feat)
        self._n_feats = len(self._header)

    @staticmethod
    def split_line(line: bytes) -> Iterable[bytes]:
        return line.split(b'\t')

    @property
    def header(self) -> Header:
        return self._header

    def sentences(self) -> Iterator[Sentence]:
        with progress_bar(total=self._corpus.file_size(), desc=self._description) as pbar:
            sentence: Sentence = []
            for n, line in enumerate(self._corpus.reader, 2):
                line = line.strip()
                if line.startswith(b'# '):
                    pbar.update(self._corpus.file_position() - pbar.n)
                    if sentence:
                        yield sentence
                        sentence = []
                elif line:
                    token = [FValue(v) for v in CSVReader.split_line(line)]
                    if len(token) < self._n_feats:
                        token += [EMPTY] * (self._n_feats - len(token))
                    assert len(token) == self._n_feats, f"Line {n}, too many columns (>{self._n_feats}): {token}"
                    sentence.append(token)
            pbar.update(self._corpus.file_position() - pbar.n)
            if sentence:
                yield sentence

    def close(self):
        self._corpus.close()


class CoNLLReader(CorpusReader):
    DEFAULT_COLUMN_HEADERS = [
        'ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'ID', 'DEPREL', 'DEPS', 'MISC'
    ]

    _corpus: CompressedFileReader
    _description: str
    _header: Header

    _n_feats: int
    _id_column: int | None
    _next_line: bytes | None

    def __init__(self, path: Path, description: str):
        self._description = description
        self._corpus = CompressedFileReader(path)

        self._next_line = self._corpus.reader.readline()
        # Try and autodetect CoNLL-U Plus with custom columns
        if match := re.match('^# global\\.columns = ([A-Z:]+(?: [A-Z:]+)+)$', self._next_line.decode()):
            header = match.group(1).split(' ')
        else:
            header = CoNLLReader.DEFAULT_COLUMN_HEADERS

        def encode(s: str) -> Feature:
            s = s.lower().replace(':', '_')
            return Feature(str.encode(s))

        self._header = [encode(s) for s in header]

        self._id_column = header.index("ID") if "ID" in header else None
        self._n_feats = len(header)

    def wordlines(self) -> Iterator[list[bytes] | None]:
        def next_wordline() -> list[bytes] | None:
            if self._next_line is None:
                self._next_line = self._corpus.reader.readline()

            if len(self._next_line) == 0:
                return None  # Reached end of file

            while self._next_line.startswith(b'#'):
                # skip comments and sentence metadata.
                self._next_line = self._corpus.reader.readline()

            # Strip linebreak and reset self._next_line so next call reads another line.
            line, self._next_line = self._next_line.rstrip(), None
            if not line:
                return []

            columns = list(CSVReader.split_line(line))
            if self._id_column is not None and not columns[self._id_column].isalnum():
                # ID might be of form X.Y (empty node) or X-Y (multiword)
                # These are not supported, so they are skipped
                return next_wordline()
            return columns

        while (wl := next_wordline()) is not None:
            yield wl

    @property
    def header(self) -> Header:
        return self._header

    def sentences(self) -> Iterator[Sentence]:
        with progress_bar(total=self._corpus.file_size(), desc=self._description) as pbar:
            sentence: Sentence = []

            for line in self.wordlines():
                if line:
                    tokens = [FValue(v) for v in line]
                    sentence.append(tokens)
                elif sentence:
                    pbar.update(self._corpus.file_position() - pbar.n)
                    yield sentence
                    sentence = []

            pbar.update(self._corpus.file_position() - pbar.n)
            if sentence:
                yield sentence

    def close(self):
        self._corpus.close()


CORPUS_READERS: dict[str, Callable[[Path, str], CorpusReader]] = {
    '.csv': CSVReader,
    '.tsv': CSVReader,
    '.conllu': CoNLLReader,
    '.conllup': CoNLLReader,
}
