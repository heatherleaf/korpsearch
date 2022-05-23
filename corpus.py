
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from disk import DiskIntArray, DiskIntArrayBuilder, DiskStringArray, DiskStringArrayBuilder
import logging

################################################################################
## Corpus

def build_corpus_index_from_csv(basedir, csv_corpusfile):
    logging.debug(f"Building corpus index...")
    corpus = open(csv_corpusfile, 'rb')

    # the first line in the CSV should be a header with the names of each column (=features)
    features = corpus.readline().decode('utf-8').split()

    with open(basedir / 'features', 'w') as features_file:
        json.dump(features, features_file)

    def words():
        # Skip over the first line
        corpus.seek(0)
        corpus.readline()

        new_sentence = True

        while True:
            line = corpus.readline()
            if not line: return

            line = line.strip()
            if line.startswith(b"# sentence"):
                new_sentence = True
            else:
                word = line.split(b'\t')
                while len(word) < len(features):
                    word.append(b'')
                yield new_sentence, word
                new_sentence = False

    strings = [set() for _feature in features]
    count = 0
    for _new_sentence, word in words():
        count += 1
        for i, feature in enumerate(word):
            strings[i].add(feature)
    logging.debug(f" --> read {sum(map(len, strings))} distinct strings")

    sentence_builder = DiskIntArrayBuilder(basedir / 'sentences',
        max_value = count-1, use_memoryview = True)
    feature_builders = []
    for i, feature in enumerate(features):
        path = basedir / ('feature.' + feature)
        builder = DiskStringArrayBuilder(path, strings[i])
        feature_builders.append(builder)

    sentence_builder.append(0) # sentence 0 doesn't exist

    sentence_count = 0
    word_count = 0
    for new_sentence, word in words():
        if new_sentence:
            sentence_builder.append(word_count)
            sentence_count += 1
        for i, feature in enumerate(word):
            feature_builders[i].append(feature)
        word_count += 1

    sentence_builder.close()
    for builder in feature_builders: builder.close()

    logging.info(f"Built corpus index, {word_count} words, {sentence_count} sentences")


class Corpus:
    dir_suffix = '.corpus'

    def __init__(self, corpus):
        basedir = Path(corpus).with_suffix(self.dir_suffix)
        self._path = Path(corpus)
        self._features = json.load(open(basedir / 'features', 'r'))
        self._features = [f.encode('utf-8') for f in self._features]
        self._sentences = DiskIntArray(basedir / 'sentences')
        self._words = \
            {feature: DiskStringArray(basedir / ('feature.' + feature.decode('utf-8')))
             for feature in self._features}
        
    def __str__(self):
        return f"[Corpus: {self._path.stem}]"

    def path(self):
        return self._path

    def strings(self, feature):
        return self._words[feature]._strings

    def intern(self, feature, value):
        return self._words[feature].intern(value)

    def num_sentences(self):
        return len(self._sentences)-1

    def sentences(self):
        for i in range(1, len(self._sentences)):
            yield self.lookup_sentence(i)

    def lookup_sentence(self, n):
        start = self._sentences[n]
        if n+1 < len(self._sentences):
            end = self._sentences[n+1]
        else:
            end = len(self._sentences)

        return [Word(self, i) for i in range(start, end)]


@dataclass(frozen=True)
class Word:
    corpus: Corpus
    pos: int

    def __getitem__(self, feature):
        return self.corpus._words[feature][self.pos]

    def keys(self):
        return self.corpus._features

    def items(self):
        for feature, value in self.corpus._words.items():
            yield feature, value[self.pos]

    def __str__(self):
        return bytes(self[b"word"]).decode('utf-8')

    def __repr__(self):
        return f"Word({dict(self.items())})"

    def __eq__(self, other):
        return dict(self) == dict(other)


def render_sentence(sentence):
    return " ".join(map(str, sentence))
