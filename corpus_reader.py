
from pathlib import Path
from collections.abc import Iterator, Callable

from util import Feature, FValue, EMPTY, SENTENCE, START, check_feature
from util import CompressedFileReader, uncompressed_suffix
from util import progress_bar, ProgressBar

Header = tuple[Feature, ...]
Token = tuple[FValue, ...]
Sentence = list[Token]
CorpusReader = tuple[Header, Iterator[Sentence]]

CORPUS_READERS: dict[str, Callable[[Path, str], CorpusReader]] = {}


def corpus_reader(path: Path, description: str) -> CorpusReader:
    suffix = uncompressed_suffix(path)
    try: 
        reader = CORPUS_READERS[suffix]
    except KeyError:
        raise ValueError(f"Cannot find a corpus reader for file type: {suffix}")
    return reader(path, description)


def csv_reader(path: Path, description: str) -> CorpusReader:
    corpus = CompressedFileReader(path)
    # the first line in the CSV should be a header with the names of each column (=features)
    header: Header = (SENTENCE,) + tuple(
        Feature(feat) for feat in corpus.reader.readline().strip().split(b'\t')
    )
    for feat in header: check_feature(feat)

    def sentence_iterator() -> Iterator[Sentence]:
        n_feats = len(header)
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
                    token: list[FValue] = [EMPTY if sentence else START]
                    token += map(FValue, line.split(b'\t'))
                    if len(token) < n_feats:
                        token += [EMPTY] * (n_feats - len(token))
                    assert len(token) == n_feats, f"Line {n}, too many columns (>{n_feats}): {token}"
                    sentence.append(tuple(token))
            pbar.update(corpus.file_position() - pbar.n)
            if sentence: 
                yield sentence

    return header, sentence_iterator()


CORPUS_READERS['.csv'] = csv_reader
CORPUS_READERS['.tsv'] = csv_reader

