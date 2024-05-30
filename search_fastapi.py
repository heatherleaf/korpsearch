
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

CORPUS_DIR = Path('corpora')

VERSION = '0.1'


setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=logging.DEBUG)

app = FastAPI()
app.mount("/webdemo", StaticFiles(directory="webdemo"), name="webdemo")


def api_call(call: Callable[..., dict[str, Any]], *args: Any, **xargs: Any) -> dict[str, Any]:
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
async def info() -> dict[str, list[Path]]:
    return api_call(get_info)

def get_info() -> dict[str, list[Path]]:
    return {
        'corpora': [
            corpus.with_suffix('')
            for corpus in CORPUS_DIR.glob('**/*.corpus')
        ]
    }


@app.get("/corpus_info")
async def corpus_info(corpus: str) -> dict[str, Any]:
    return api_call(get_corpus_info, corpus)

def get_corpus_info(corpus_path: Path) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    with Corpus(corpus_path) as corpus:
        indexes: list[str] = []
        for index_path in corpus_path.with_suffix('.indexes').glob('*:*'):
            templ = Template.parse(corpus, index_path.name)
            indexes.append(templ.querystr())
        return {
            'info': {
                'id': corpus_path,
                'name': corpus_path.stem,
                'description': '?',
                'size': len(corpus),
                'sentences': corpus.num_sentences(),
                'features': corpus.features,
                'indexes': indexes,
            }
        }


@app.get("/search")
async def search(
        corpus: str,
        cqp: str, 
        start: int = 0,
        num: int = 10,
        end: int = -1,
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
            num = num,
            end = end,
            show = show, 
            filter = filter,
            no_cache = no_cache, 
            no_sentence_breaks = no_sentence_breaks,
            internal_intersection = internal_intersection,
        )
    )

