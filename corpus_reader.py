
from pathlib import Path
from collections.abc import Iterator, Callable

from util import Feature, FValue, EMPTY, SENTENCE, START, check_feature
from util import CompressedFileReader, uncompressed_suffix
from util import progress_bar, ProgressBar

Header = list[Feature]
Token = list[FValue]
Sentence = list[Token]
CorpusReader = tuple[Header, Iterator[Sentence]]

CORPUS_READERS: dict[str, Callable[[Path, str], CorpusReader]] = {}


def corpus_reader(path: Path, description: str, sentence_feature: bool = True, 
                  reversed_features: bool = True) -> CorpusReader:
    suffix = uncompressed_suffix(path)
    try: 
        reader = CORPUS_READERS[suffix]
    except KeyError:
        raise ValueError(f"Cannot find a corpus reader for file type: {suffix}")
    return augment_reader(reader(path, description), sentence_feature, reversed_features)


def augment_reader(reader: CorpusReader, sentence_feature: bool, reversed_features: bool) -> CorpusReader:
    header, sentences = reader
    header = list(header)  # Make a copy of header, so it won't interfere with the sentence iterator
    if reversed_features:
        header += [Feature(feature + b'_rev') for feature in header]
    if sentence_feature:
        header.append(SENTENCE)
    return header, augment_sentences(sentences, sentence_feature, reversed_features)


def augment_sentences(sentence_iterator: Iterator[Sentence], sentence_feature: bool,
                      reversed_features: bool) -> Iterator[Sentence]:
    for sentence in sentence_iterator:
        for token in sentence:
            if reversed_features:
                token += [FValue(val.decode()[::-1].encode()) for val in token]
            if sentence_feature:
                token.append(EMPTY)
        if sentence_feature:
            sentence[0][-1] = START
        yield sentence
            


def csv_reader(path: Path, description: str) -> CorpusReader:
    corpus = CompressedFileReader(path)
    # the first line in the CSV should be a header with the names of each column (=features)
    header: Header = list(map(Feature, corpus.reader.readline().strip().split(b'\t')))
    for feat in header: check_feature(feat)
    n_feats = len(header)

    def sentence_iterator() -> Iterator[Sentence]:
        pbar: ProgressBar[None]
        with progress_bar(total=corpus.file_size(), desc=description) as pbar:
            sentence: Sentence = []
            for n, line in enumerate(corpus.reader, 2):
                line = line.strip()
                if line.startswith(b'# '):
                    pbar.update(corpus.file_position() - pbar.n)
                    if sentence: 
                        yield sentence
                        sentence = []
                elif line:
                    token = list(map(FValue, line.split(b'\t')))
                    if len(token) < n_feats:
                        token += [EMPTY] * (n_feats - len(token))
                    assert len(token) == n_feats, f"Line {n}, too many columns (>{n_feats}): {token}"
                    sentence.append(token)
            pbar.update(corpus.file_position() - pbar.n)
            if sentence: 
                yield sentence

    return header, sentence_iterator()


CORPUS_READERS['.csv'] = csv_reader
CORPUS_READERS['.tsv'] = csv_reader

