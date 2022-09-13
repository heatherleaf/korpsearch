
import json
from pathlib import Path
import logging
from typing import BinaryIO, List, Tuple, Set, Dict, Iterator, Sequence

import disk
from util import progress_bar

################################################################################
## Corpus

def build_corpus_index_from_csv(basedir:Path, csv_corpusfile:Path):
    logging.debug(f"Building corpus index...")
    corpus : BinaryIO = open(csv_corpusfile, 'rb')
    corpus.seek(0, 2)
    csv_filesize = corpus.tell()

    # the first line in the CSV should be a header with the names of each column (=features)
    corpus.seek(0)
    features : List[str] = corpus.readline().decode().split()

    with open(basedir / Corpus.features_file, 'w') as OUT:
        json.dump(features, OUT)

    def iterate_words(description) -> Iterator[Tuple[bool, List[bytes]]]:
        # Skip over the first line
        corpus.seek(0)
        corpus.readline()

        with progress_bar(total=csv_filesize, desc=description) as pbar:
            new_sentence : bool = True
            for line in corpus:
                pbar.update(len(line))
                line = line.strip()
                if line.startswith(b'# '):
                    new_sentence = True
                elif line:
                    word : List[bytes] = line.split(b'\t')
                    if len(word) < len(features):
                        word += [b''] * (len(features) - len(word))
                    yield new_sentence, word
                    new_sentence = False

    strings : List[Set[bytes]] = [set() for _feature in features]
    count : int = 0
    for _new_sentence, word in iterate_words("Collecting strings"):
        count += 1
        for i, feat in enumerate(word):
            strings[i].add(feat)
    logging.debug(f" --> read {sum(map(len, strings))} distinct strings")

    sentence_builder = disk.DiskIntArrayBuilder(
        basedir / Corpus.sentences_path, 
        max_value=count-1, 
        use_memoryview=True,
    )
    feature_builders : List[disk.DiskStringArrayBuilder] = []
    for i, feature in enumerate(features):
        path = basedir / (Corpus.feature_prefix + feature) / feature
        path.parent.mkdir(exist_ok=True)
        builder = disk.DiskStringArrayBuilder(path, strings[i])
        feature_builders.append(builder)

    sentence_builder.append(0) # sentence 0 doesn't exist

    sentence_count : int = 0
    word_count : int = 0
    for new_sentence, word in iterate_words("Building indexes"):
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
    features_file = 'features.cfg'
    feature_prefix = 'feature:'
    sentences_path = 'sentences'

    features : List[str]
    words : Dict[str, disk.DiskStringArray]
    sentence_pointers : disk.DiskIntArrayType
    path : Path

    def __init__(self, corpus:Path):
        basedir : Path = corpus.with_suffix(self.dir_suffix)
        self.path = corpus
        with open(basedir / self.features_file, 'r') as IN:
            self.features = json.load(IN)
        self.sentence_pointers = disk.DiskIntArray(basedir / self.sentences_path)
        self.words = {
            feature: disk.DiskStringArray(basedir / (self.feature_prefix + feature) / feature)
            for feature in self.features
        }
        assert all(
            len(self) == len(arr) for arr in self.words.values()
        )
        
    def __str__(self) -> str:
        return f"[Corpus: {self.path.stem}]"

    def __len__(self) -> int:
        return len(self.words[self.features[0]])

    def strings(self, feature:str) -> disk.StringCollection:
        return self.words[feature].strings

    def intern(self, feature:str, value:bytes) -> disk.InternedString:
        return self.words[feature].intern(value)

    def num_sentences(self) -> int:
        return len(self.sentence_pointers)-1

    def sentences(self) -> Iterator[slice]:
        sents : disk.DiskIntArrayType = self.sentence_pointers
        for start, end in zip(sents[1:], sents[2:]):
            yield slice(start, end)
        yield slice(sents[-1], len(self))

    def lookup_sentence(self, n:int) -> slice:
        sents : disk.DiskIntArrayType = self.sentence_pointers
        start : int = sents[n]
        nsents : int = len(sents)
        end : int = sents[n+1] if n+1 < nsents else nsents
        return slice(start, end)

    def render_sentence(self, sent:int, features_to_show:Sequence[str]=()) -> str:
        if not features_to_show:
            features_to_show = self.features[:1]
        positions = self.lookup_sentence(sent)
        return ' '.join(
            '/'.join(str(self.words[feat][pos]) for feat in features_to_show) 
            for pos in range(positions.start, positions.stop)
        )

    def get_sentence_from_token(self, pos:int) -> int:
        ptrs : disk.DiskIntArrayType = self.sentence_pointers
        start : int = 0
        end : int = len(ptrs) - 1
        while start <= end:
            mid = (start + end) // 2
            if ptrs[mid] <= pos:
                start = mid + 1
            else:
                end = mid - 1
        return end

