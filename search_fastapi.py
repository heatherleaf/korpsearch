
import logging
from time import time
from pathlib import Path
from argparse import Namespace
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Any
from collections.abc import Callable

from corpus import Corpus
from index import Template
from util import setup_logger
from search import main_search

VERSION = '0.3'

CORPUS_DIR = Path('corpora')
CHARSET = 'utf8'

setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=logging.DEBUG)

app = FastAPI()
app.mount("/webdemo", StaticFiles(directory="webdemo"), name="webdemo")

APIResult = dict[str, Any]


def api_call(call: Callable[..., dict[str, Any]], *args: Any, **xargs: Any) -> APIResult:
    start_time = time()
    try:
        result = call(*args, **xargs)
        result['time'] = time() - start_time
        return result
    except FileNotFoundError as e:
        error_code = 404
        error_detail = str(e)
    except Exception as e:
        error_code = 500
        error_detail = str(e)
    raise HTTPException(status_code=error_code, detail=error_detail)


@app.get("/info")
async def info() -> APIResult:
    return api_call(get_info)

def get_info() -> APIResult:
    return {
        'corpora': [
            corpus.with_suffix('')
            for corpus in CORPUS_DIR.glob('**/*.corpus')
        ],
        'version': VERSION,
    }


@app.get("/corpus_info")
async def corpus_info(corpus: str) -> APIResult:
    return api_call(get_corpus_info, corpus.split(','))

def get_corpus_info(corpus_paths: list[str]) -> APIResult:
    result: APIResult = {}
    for path in corpus_paths:
        corpus_path = Path(path.strip())
        with Corpus(corpus_path) as corpus:
            indexes: list[str] = []
            for index_path in corpus.path.with_suffix('.indexes').glob('*:*'):
                templ = Template.parse(corpus, index_path.name)
                indexes.append(templ.querystr())
            features = [feat.decode() for feat in corpus.features]
            tokens = len(corpus)
            sentences = corpus.num_sentences()
            result[corpus.id] = {
                'attrs': {
                    'p': features,  # positional attributes
                    's': [],        # structural attributes
                    'a': [],        # aligned attributes, for linked corpora
                },
                'info': {
                    'Name': corpus.path.stem,
                    'Charset': CHARSET,
                    'Size': tokens,
                    'Sentences': sentences,
                    'Indexes': indexes,
                }
            }
    return {
        'corpora': result,
        'total_size': sum(c['info']['Size'] for c in result.values()),
        'total_sentences': sum(c['info']['Sentences'] for c in result.values()),
    }


@app.get("/query")
async def query(
        corpus: str,
        cqp: str,
        start: int = 0,
        num: int = 10,
        end: int = -1,
        show: str = "",
        filter: bool = False,
        no_cache: bool = False,
        no_diskarray: bool = False,
        no_binary: bool = False,
        no_sentence_breaks: bool = False,
        internal_merge: bool = False,
    ) -> APIResult:
    return api_call(
        main_search,
        Namespace(
            corpus = corpus,
            query = cqp,
            start = start,
            num = num,
            end = end,
            show = show,
            filter = filter,
            no_cache = no_cache,
            no_diskarray = no_diskarray,
            no_binary = no_binary,
            no_sentence_breaks = no_sentence_breaks,
            internal_merge = internal_merge,
        )
    )

