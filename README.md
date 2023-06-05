
# Korp-search

Searching in very large corpora.

The basic idea is to use [inverted indexes](https://en.wikipedia.org/wiki/Inverted_index) to lookup queries. The indexes are modifications of [suffix arrays](https://en.wikipedia.org/wiki/Suffix_array), but where we don't use the whole suffix -- so a "pruned" suffix array if you like. We build two kinds of indexes -- unary (for looking up one single feature), and binary (for looking up a pair of adjacent features). A complex search query is then translated to a conjunction of simple queries which use unary or binary indexes. Then we calculate the intersection of all query results.

## Dependencies

There are no required libraries, you should be able to run things without installing anything extra. However, we recommend that you install the following:

- [tqdm](https://pypi.org/project/tqdm/) for a better progress bar

- [Cython](https://pypi.org/project/cython/) if you want to use the fast intersection algorithm (see below)

- [FastAPI](https://pypi.org/project/fastapi/) if you want to run the web demo (see below)

## Building inverted indexes

Before you can start searching in a corpus you have to build the corpus index, and then some inverted indexes. If you want help, run:
```
python build_indexes.py --help
```

The following builds the basic corpus index for the BNC-mini corpus, in the directory `corpora/bnc-mini.corpus/`, which is a compact and efficient representation of the corpus:
```
python build_indexes.py --corpus corpora/bnc-mini.csv --corpus-index
```

Now you can build inverted indexes for the BNC-mini corpus, for the features `word`, `lemma`, and `pos`. All inverted indexes are in the directory `corpora/bnc-mini.indexes/`:
```
python build_indexes.py --corpus corpora/bnc-mini.csv --features word lemma pos --max-dist 2
```

`--max-dist` tells how many different binary indexes that will be created: it's the maximum adjacent distance between the tokens in the query. The default setting is 2. If you only want unary indexes you can set `--max-dist` to 0, and if you want mroe control you can use the `--templates` option instead of `--features`.

The original `.csv` file is not used when searching, so you can remove it if you want (but then you cannot build any new indexes, so it's probably a bad idea).

## Searching from the command line

To search from the command line, you give the corpus and the query. Note that you have to have built the inverted indexes as described above:
```
python search_cmdline.py --corpus corpora/bnc-mini --query '[pos="ART"] [lemma="small"] [pos="SUBST"]'
```

All searches are cached (in the directory `cache/`), if you don't want to use the cached result, you can use `--no-cache`. Use `--print json` for JSON output, `--start 42` to print from match 42, and `--num 14` to print at most 14 matches. Use the following for more help: 
```
python search_cmdline.py --help
```

## Using the web demo

To use the web demo you need to install [FastAPI](https://pypi.org/project/fastapi/). You also have to build corpus- and search indexes for the corpora you want to play with. The you can run the following to start the webserver:
```
uvicorn search_fastapi:app --reload
```

When the server has started you can open the URL <http://127.0.0.1:8000/webdemo/index.html> to play around with the web demo.
