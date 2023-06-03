
import logging
from time import time
from pathlib import Path
from argparse import Namespace
from fastapi import FastAPI, HTTPException
from typing import List

from corpus import Corpus
from index import Index, Template, TemplateLiteral
from util import setup_logger
from search import main_search

CORPUS_DIRS = [
    'corpora',
    'private',
]

VERSION = '0.1'


setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=logging.DEBUG)

app = FastAPI()

def api_call(call, *args, **xargs):
    start_time = time()
    try: 
        result = call(*args, **xargs)
        result['time'] = time() - start_time
        result['version'] = VERSION
        return result
    except FileNotFoundError as e:
        error_code = 404
        error_detail = str(e)
    except Exception as e:
        error_code = 500
        error_detail = str(e)
    raise HTTPException(status_code=error_code, detail=error_detail)


@app.get("/info")
async def info():
    return api_call(get_info)

def get_info():
    return {
        'corpora': [
            corpus.with_suffix('')
            for dir in CORPUS_DIRS
            for corpus in Path(dir).glob('**/*.corpus')
        ]
    }


@app.get("/corpus_info")
async def corpus_info(corpus: str):
    return api_call(get_corpus_info, corpus)

def get_corpus_info(corpus_path):
    corpus_path = Path(corpus_path)
    with Corpus(corpus_path) as corpus:
        indexes = []
        for feat in corpus.features:
            try:
                Index(corpus, Template([TemplateLiteral(0, feat)]))
            except FileNotFoundError:
                continue
            indexes.append(feat)
        return {
            'info': {
                'id': corpus_path,
                'name': corpus_path.stem,
                'description': '?',
                'size': len(corpus),
                'sentences': corpus.num_sentences(),
                'features': corpus.features,
                'indexed-features': indexes,
            }
        }


@app.get("/search")
async def search(
        corpus: str,
        cqp: str, 
        start: int = 0,
        end: int = 9,
        show: str = "", 
        filter: bool = False,
        no_cache: bool = False, 
        no_sentence_breaks: bool = False,
        internal_intersection: bool = False,
    ):
    return api_call(
        main_search,
        Namespace(
            corpus = corpus,
            query = cqp, 
            start = start,
            end = end,
            show = show, 
            filter = filter,
            no_cache = no_cache, 
            no_sentence_breaks = no_sentence_breaks,
            internal_intersection = internal_intersection,
        )
    )

