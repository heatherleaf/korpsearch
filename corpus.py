
import json
from pathlib import Path
from dataclasses import dataclass
import disk
from typing import BinaryIO, List, Tuple, Set, Dict, Iterator
import logging

################################################################################
## Corpus

def build_corpus_index_from_csv(basedir:Path, csv_corpusfile:Path):
    logging.debug(f"Building corpus index...")
    corpus : BinaryIO = open(csv_corpusfile, 'rb')

    # the first line in the CSV should be a header with the names of each column (=features)
    features : List[str] = corpus.readline().decode('utf-8').split()

    with open(basedir / 'features', 'w') as features_file:
        json.dump(features, features_file)

    def words() -> Iterator[Tuple[bool, List[bytes]]]:
        # Skip over the first line
        corpus.seek(0)
        corpus.readline()

        new_sentence : bool = True

        while True:
            line : bytes = corpus.readline()
            if not line: return

            line = line.strip()
            if line.startswith(b"# sentence"):
                new_sentence = True
            else:
                word : List[bytes] = line.split(b'\t')
                while len(word) < len(features):
                    word.append(b'')
                yield new_sentence, word
                new_sentence = False

    strings : List[Set[bytes]] = [set() for _feature in features]
    count : int = 0
    for _new_sentence, word in words():
        count += 1
        for i, feat in enumerate(word):
            strings[i].add(feat)
    logging.debug(f" --> read {sum(map(len, strings))} distinct strings")

    sentence_builder = disk.DiskIntArrayBuilder(
        basedir/'sentences', max_value=count-1, use_memoryview=True)
    feature_builders : List[disk.DiskStringArrayBuilder] = []
    for i, feature in enumerate(features):
        path = basedir / ('feature.' + feature)
        builder = disk.DiskStringArrayBuilder(path, strings[i])
        feature_builders.append(builder)

    sentence_builder.append(0) # sentence 0 doesn't exist

    sentence_count : int = 0
    word_count : int = 0
    for new_sentence, word in words():
        if new_sentence:
            sentence_builder.append(word_count)
            sentence_count += 1
        for i, feat in enumerate(word):
            feature_builders[i].append(feat)
        word_count += 1

    # sentence_builder.close()
    # for builder in feature_builders: builder.close()

    logging.info(f"Built corpus index, {word_count} words, {sentence_count} sentences")


class Corpus:
    dir_suffix = '.corpus'

    features : List[bytes]
    words : Dict[bytes, disk.DiskStringArray]
    _sentences : disk.DiskIntArrayType
    _path : Path

    def __init__(self, corpus:Path):
        basedir : Path = corpus.with_suffix(self.dir_suffix)
        self._path = corpus
        self.features = [f.encode('utf-8') for f in json.load(open(basedir/'features', 'r'))]
        self._sentences = disk.DiskIntArray(basedir / 'sentences')
        self.words = {
            feature: disk.DiskStringArray(basedir / ('feature.' + feature.decode('utf-8')))
            for feature in self.features
        }
        
    def __str__(self) -> str:
        return f"[Corpus: {self._path.stem}]"

    def path(self) -> Path:
        return self._path

    def strings(self, feature:bytes) -> disk.StringCollection:
        return self.words[feature].strings

    def intern(self, feature:bytes, value:bytes) -> disk.InternedString:
        return self.words[feature].intern(value)

    def num_sentences(self) -> int:
        return len(self._sentences)-1

    def sentences(self) -> Iterator[List['Word']]:
        for i in range(1, len(self._sentences)):
            yield self.lookup_sentence(i)

    def lookup_sentence(self, n:int) -> List['Word']:
        start : int = self._sentences[n]
        nsents : int = len(self._sentences)
        end : int = self._sentences[n+1] if n+1 < nsents else nsents
        return [Word(self, i) for i in range(start, end)]


@dataclass(frozen=True)
class Word:
    corpus: Corpus
    pos: int

    def __getitem__(self, feature:bytes) -> disk.InternedString:
        return self.corpus.words[feature][self.pos]

    def keys(self) -> List[bytes]:
        return self.corpus.features

    def items(self) -> Iterator[Tuple[bytes, disk.InternedString]]:
        for feature, value in self.corpus.words.items():
            yield feature, value[self.pos]

    def __str__(self) -> str:
        return bytes(self[b"word"]).decode('utf-8')

    def __repr__(self) -> str:
        return f"Word({dict(self.items())})"

    def __eq__(self, other:object) -> bool:
        if isinstance(other, Word):
            return dict(self) == dict(other)
        return False


def render_sentence(sentence:List[Word]) -> str:
    return " ".join(map(str, sentence))
