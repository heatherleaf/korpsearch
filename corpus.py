
import json
from pathlib import Path
import logging
from typing import BinaryIO, List, Set, Dict, Iterator, Sequence, Union
from types import TracebackType

from disk import DiskIntArray, DiskIntArrayBuilder, DiskStringArray, DiskStringArrayBuilder, StringCollection, InternedString
from util import progress_bar, CompressedFileReader, add_suffix

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

    features : List[str]
    tokens : Dict[str, DiskStringArray]
    sentence_pointers : DiskIntArray
    path : Path

    def __init__(self, corpus:Union[Path,str]):
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

    def strings(self, feature:str) -> StringCollection:
        return self.tokens[feature].strings

    def intern(self, feature:str, value:bytes, is_prefix = False) -> List[InternedString]:
        return self.tokens[feature].intern(value, is_prefix)

    def num_sentences(self) -> int:
        return len(self.sentence_pointers)-1

    def sentences(self) -> Iterator[range]:
        sents = self.sentence_pointers
        for start, end in zip(sents[1:], sents[2:]):
            yield range(start, end)
        yield range(sents[len(sents)-1], len(self))

    def sentence_positions(self, n:int) -> range:
        sents = self.sentence_pointers
        start = sents[n]
        nsents = len(sents)
        end = sents[n+1] if n+1 < nsents else nsents
        return range(start, end)

    def render_sentence(self, sent:int, pos:int=-1, offset:int=-1, features:Sequence[str]=(), context:int=-1) -> str:
        if not features:
            features = self.features[:1]
        tokens = []
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

    def get_sentence_from_position(self, pos:int) -> int:
        ptrs = self.sentence_pointers
        start, end = 0, len(ptrs)-1
        while start <= end:
            mid = (start + end) // 2
            if ptrs[mid] <= pos:
                start = mid + 1
            else:
                end = mid - 1
        return end

    def __enter__(self) -> 'Corpus':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        for sa in self.tokens.values(): sa.close()
        self.sentence_pointers.close()

    @staticmethod
    def build_from_csv(basedir:Path, csv_corpusfile:Path):
        logging.debug(f"Building corpus index")
        corpus = CompressedFileReader(csv_corpusfile)
        csv_filesize = corpus.file_size()

        # the first line in the CSV should be a header with the names of each column (=features)
        corpus.reader.seek(0)
        features : List[str] = corpus.reader.readline().decode().split()
        assert Corpus.sentence_feature not in features
        features.insert(0, Corpus.sentence_feature)

        with open(basedir / Corpus.features_file, 'w') as OUT:
            json.dump(features, OUT)

        def iterate_sentences(description:str) -> Iterator[List[List[bytes]]]:
            # Skip over the header line
            corpus.reader.seek(0)
            corpus.reader.readline()
            with progress_bar(total=csv_filesize, desc=description) as pbar:
                sentence = []
                for line in corpus.reader:
                    pbar.update(corpus.file_position() - pbar.n)
                    line = line.strip()
                    if line.startswith(b'# '):
                        if sentence: 
                            yield sentence
                            sentence = []
                    elif line:
                        token : List[bytes] = line.split(b'\t')
                        token.insert(0, Corpus.empty_value if sentence else Corpus.sentence_start_value)
                        if len(token) < len(features):
                            token += [Corpus.empty_value] * (len(features) - len(token))
                        sentence.append(token)
                if sentence: 
                    yield sentence

        strings : List[Set[bytes]] = [set() for _feature in features]
        n_sentences = n_tokens = 0
        for sentence in iterate_sentences("Collecting strings"):
            n_sentences += 1
            for token in sentence:
                n_tokens += 1
                for i, feat in enumerate(token):
                    strings[i].add(feat)
        logging.debug(f" --> read {sum(map(len, strings))} distinct strings, {n_sentences} sentences, {n_tokens} tokens")

        sentence_builder = DiskIntArrayBuilder(basedir / Corpus.sentences_path, max_value=n_tokens)
        sentence_builder.append(0) # sentence 0 doesn't exist

        feature_builders : List[DiskStringArrayBuilder] = []
        for i, feature in enumerate(features):
            path = basedir / (Corpus.feature_prefix + feature) / feature
            path.parent.mkdir(exist_ok=True)
            builder = DiskStringArrayBuilder(path, strings[i])
            feature_builders.append(builder)

        ctr = 0
        for sentence in iterate_sentences("Building indexes"):
            sentence_builder.append(ctr)
            for token in sentence:
                ctr += 1
                for i, feat in enumerate(token):
                    feature_builders[i].append(feat)
        assert ctr == n_tokens

        sentence_builder.close()
        for builder in feature_builders: 
            builder.close()

        logging.info(f"Built corpus index, {n_tokens} tokens, {n_sentences} sentences")
