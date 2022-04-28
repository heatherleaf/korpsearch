import os
import time
import json
import shutil
from pathlib import Path
from disk import *

def log(output, verbose, start=None):
    if verbose:
        if start:
            duration = time.time()-start
            print(output.ljust(100), f"{duration//60:4.0f}:{duration%60:05.2f}")
        else:
            print(output)

################################################################################
## Corpus

def build_corpus_index(corpusfile, verbose=False):
    log(f"Building corpus index", verbose)
    basedir = Path(corpusfile).with_suffix('.corpus')
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)
    corpus = open(Path(corpusfile).with_suffix('.csv'), 'rb')
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

    t0 = time.time()
    strings = [set() for _feature in features]
    count = 0
    for _new_sentence, word in words():
        count += 1
        for i, feature in enumerate(word):
            strings[i].add(feature)
    log(f" -> interned {sum(map(len, strings))} strings", verbose, start=t0)

    t0 = time.time()
    sentence_builder = DiskIntArrayBuilder(basedir / 'sentences',
        max_value = count-1, use_mmap = True)
    feature_builders = []
    for i, feature in enumerate(features):
        path = basedir / ('feature.' + feature)
        builder = DiskStringArrayBuilder(path, strings[i])
        feature_builders.append(builder)

    sentence_builder.append(0) # sentence 0 doesn't exist

    sentence_count = 0
    ctr = 0
    for new_sentence, word in words():
        if new_sentence:
            sentence_builder.append(ctr)
            sentence_count += 1
        for i, feature in enumerate(word):
            feature_builders[i].append(feature)
        ctr += 1

    sentence_builder.close()
    for builder in feature_builders: builder.close()

    log(f" -> built corpus index, {ctr} words, {sentence_count} sentences", verbose, start=t0)
    log("", verbose)

class Corpus:
    def __init__(self, corpus):
        basedir = Path(corpus).with_suffix('.corpus')
        self._path = Path(corpus)
        self._features = json.load(open(basedir / 'features', 'r'))
        self._features = [f.encode('utf-8') for f in self._features]
        self._sentences = DiskIntArray(basedir / 'sentences')
        self._words = \
            {feature: DiskStringArray(basedir / ('feature.' + feature.decode('utf-8')))
             for feature in self._features}
        
    def __str__(self):
        return f"[Corpus: {self._path.stem}]"

    def sentences(self):
        for i in range(1, len(self._sentences)):
            yield self.lookup_sentence(i)

    def lookup_sentence(self, n):
        start = self._sentences[n]
        if n+1 < len(self._sentences):
            end = self._sentences[n+1]
        else:
            end = len(self._sentences)

        return [self._get_word(i) for i in range(start, end)]

    def _get_word(self, i):
        return {feature: words[i]
                for feature, words in self._words.items()}
