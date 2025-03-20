
import logging
from time import time
from pathlib import Path
from argparse import Namespace, ArgumentParser
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
from collections.abc import Callable

from corpus import Corpus
from index import Template
from util import setup_logger
from search import main_search

SETTINGS = Namespace(
    version = '0.3',
    base_dir = Path('corpora'),
    cache_dir = Path('cache'),
    demo_dir = Path('webdemo'),
    host = '127.0.0.1',
    port = 8000,
    ssl_keyfile = None,
    ssl_certfile = None,
    charset = 'utf8',
    loglevel = 'info',
    filter = False,
    no_cache = False,
    no_diskarray = False,
    no_binary = False,
    no_sentence_breaks = False,
    internal_merge = False,
)

setup_logger(
    '{warningname}  {message}',
    timedivider = 1000,
    loglevel = getattr(logging, SETTINGS.loglevel.upper()),
)


def get_corpora() -> list[str]:
    return [corpus.stem for corpus in SETTINGS.base_dir.glob('*' + Corpus.dir_suffix)]


################################################################################
## API

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.mount(f"/{SETTINGS.demo_dir}", StaticFiles(directory=SETTINGS.demo_dir), name=str(SETTINGS.demo_dir))

APIResult = dict[str, Any]


def api_call(call: Callable[..., dict[str, Any]], *args: Any, **xargs: Any) -> APIResult:
    start_time = time()
    try:
        result = call(*args, **xargs)
        result['time'] = time() - start_time
        callargs = [f"{a!r}" for a in args] + [f"{k}={v!r}" for k,v in xargs.items()]
        logging.info(f"Completed in {result['time']:.3f} s: {call.__name__}({', '.join(callargs)})")
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
        'corpora': get_corpora(),
        'version': SETTINGS.version,
    }


@app.get("/corpus_info")
async def corpus_info(corpus: str) -> APIResult:
    return api_call(get_corpus_info, corpus)

def get_corpus_info(corpora: str) -> APIResult:
    result: APIResult = {}
    for cid in corpora.split(','):
        with Corpus(cid, base_dir=SETTINGS.base_dir) as corpus:
            indexes: list[str] = []
            for index_path in corpus.path.with_suffix('.indexes').glob('*:*'):
                templ = Template.parse(corpus, index_path.name)
                indexes.append(templ.querystr())
            features = [feat.decode() for feat in corpus.features]
            tokens = len(corpus)
            sentences = corpus.num_sentences()
            result[corpus.name] = {
                'attrs': {
                    'p': features,  # positional attributes
                    's': [],        # structural attributes
                    'a': [],        # aligned attributes, for linked corpora
                },
                'info': {
                    'Name': corpus.path.stem,
                    'Charset': SETTINGS.charset,
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
        filter: bool|None = None,
        no_cache: bool|None = None,
        no_diskarray: bool|None = None,
        no_binary: bool|None = None,
        no_sentence_breaks: bool|None = None,
        internal_merge: bool|None = None,
    ) -> APIResult:
    return api_call(
        run_query,
        corpus,
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

def run_query(corpus: str, **xargs: Any) -> APIResult:
    args = SETTINGS.__dict__
    args['corpus'] = corpus.split(',')
    for k, v in xargs.items():
        if v is not None: args[k] = v
    return main_search(Namespace(**args))


################################################################################
## Main

parser = ArgumentParser(description='Search API')
parser.add_argument('--base-dir', '-d', type=Path, help=f'directory where to find the corpora (default: {SETTINGS.base_dir})')
parser.add_argument('--cache-dir', type=Path, help=f'directory where to store cache files (default: {SETTINGS.cache_dir})')
parser.add_argument('--host', type=str, help=f'host name to use (default: {SETTINGS.host})')
parser.add_argument('--port', type=int, help=f'port number to use (default: {SETTINGS.port})')
parser.add_argument('--ssl-keyfile', type=str, help=f'SSL key file')
parser.add_argument('--ssl-certfile', type=str, help=f'SS certificate file')

parser.add_argument('--no-cache', action="store_true", help="don't use cached queries")
parser.add_argument('--no-diskarray', action="store_true", help="don't use on-disk arrays")
parser.add_argument('--no-binary', action="store_true", help="don't use binary indexes")
parser.add_argument('--internal-merge', action='store_true', help='use the internal (slow) merge')
parser.add_argument('--no-sentence-breaks', action='store_true', help="don't care about sentence breaks")
parser.add_argument('--filter', action='store_true', help='filter the final results (should not be necessary)')

parser.add_argument('--quiet', action="store_const", dest="loglevel", const='warning', help='quiet mode')
parser.add_argument('--debug', action="store_const", dest="loglevel", const='debug', help='debugging mode')

if __name__ == "__main__":
    parser.parse_args(namespace=SETTINGS)
    logging.getLogger().setLevel(getattr(logging, SETTINGS.loglevel.upper()))
    print(f"Open web demo here: http://{SETTINGS.host}:{SETTINGS.port}/{SETTINGS.demo_dir}/index.html")
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level=SETTINGS.loglevel,
                ssl_keyfile=SETTINGS.ssl_keyfile, ssl_certfile=SETTINGS.ssl_certfile)
