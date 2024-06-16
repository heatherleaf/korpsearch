
from pathlib import Path
from collections.abc import Iterator, Callable

from util import EMPTY, SENTENCE, check_feature
from util import CompressedFileReader, uncompressed_suffix
from util import progress_bar, ProgressBar

Token = tuple[bytes, ...]
Sentence = list[Token]

CORPUS_READERS: dict[str, Callable[[Path, str], Iterator[Sentence]]] = {}


def corpus_reader(path: Path, description: str) -> tuple[Token, Iterator[Sentence]]:
    suffix = uncompressed_suffix(path)
    try: 
        reader = CORPUS_READERS[suffix]
        sentences = reader(path, description)
    except KeyError:
        raise ValueError(f"Cannot find a corpus reader for file type: {suffix}")
    header_sentence = next(sentences)
    return header_sentence[0], sentences


def csv_reader(path: Path, description: str) -> Iterator[Sentence]:
    corpus = CompressedFileReader(path)
    with corpus as linereader:
        # the first line in the CSV should be a header with the names of each column (=features)
        header = (SENTENCE,) + tuple(linereader.readline().strip().split(b'\t'))
        for feat in header: check_feature(feat)
        yield [header]

        n_feats = len(header)
        pbar: ProgressBar[None]
        with progress_bar(total=corpus.file_size(), desc=description) as pbar:
            sentence: Sentence = []
            for line in linereader:
                line = line.strip()
                if line.startswith(b'# '):
                    pbar.update(corpus.file_position() - pbar.n)
                    if sentence: 
                        yield sentence
                        sentence = []
                elif line:
                    token = [EMPTY if sentence else SENTENCE]
                    token += line.split(b'\t')
                    if len(token) < n_feats:
                        token += [EMPTY] * (n_feats - len(token))
                    sentence.append(tuple(token))
            pbar.update(corpus.file_position() - pbar.n)
            if sentence: 
                yield sentence

CORPUS_READERS['.csv'] = csv_reader
CORPUS_READERS['.tsv'] = csv_reader

