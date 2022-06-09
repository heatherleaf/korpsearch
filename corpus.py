
import json
from pathlib import Path
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

    sentence_builder.close()
    for builder in feature_builders: 
        builder.close()

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

    def sentences(self) -> Iterator[slice]:
        sents : disk.DiskIntArrayType = self._sentences
        for start, end in zip(sents[1:], sents[2:]):
            yield slice(start, end)
        yield slice(sents[-1], len(sents))

    def lookup_sentence(self, n:int) -> slice:
        sents : disk.DiskIntArrayType = self._sentences
        start : int = sents[n]
        nsents : int = len(sents)
        end : int = sents[n+1] if n+1 < nsents else nsents
        return slice(start, end)

    def render_sentence(self, n:int) -> str:
        # TODO: the feature(s) to show should be configurable
        feat : bytes = b'word' if b'word' in self.features else self.features[0]
        words : disk.DiskStringArray = self.words[feat]
        sent : slice = self.lookup_sentence(n)
        return " ".join(map(str, words[sent]))

