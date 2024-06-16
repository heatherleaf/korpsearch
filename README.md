
# Korp-search

Searching in very large corpora.

The basic idea is to use [inverted indexes](https://en.wikipedia.org/wiki/Inverted_index) to lookup queries. 
The indexes are modifications of [suffix arrays](https://en.wikipedia.org/wiki/Suffix_array), but where we don't use the whole suffix -- so a "pruned" suffix array if you like. 
We build two kinds of indexes -- unary (for looking up one single feature), and binary (for looking up a pair of adjacent features). 
A complex search query is then translated to a conjunction of simple queries which use unary or binary indexes. 
Then we calculate the intersection of all query results.

## Dependencies

There are no required libraries, you should be able to run things without installing anything extra. 
However, we recommend that you install the following:

- [tqdm](https://pypi.org/project/tqdm/) for a better progress bar

- [Cython](https://pypi.org/project/cython/) if you want to use the fast merging algorithm (see below)

- [FastAPI](https://pypi.org/project/fastapi/) if you want to run the web demo (see below)

- [PyPy](https://www.pypy.org/) if you want building the indexes to be around twice as fast
  (Note: using PyPy doesn't seem to improve the search efficiency)

## Building inverted indexes

Before you can start searching in a corpus you have to build the corpus index, and then some inverted indexes. If you want help, run:

    python3 build_indexes.py --help

The following builds the basic corpus index for the smallest version of the BNC corpus, in the directory `corpora/bnc-100k.corpus/`, which is a compact and efficient representation of the corpus:

    python3 build_indexes.py --corpus corpora/bnc-100k.csv --corpus-index

Now you can build inverted indexes for the corpus, for the features `word`, `lemma`, and `pos`. All inverted indexes are in the directory `corpora/bnc-100k.indexes/`:

    python3 build_indexes.py --corpus corpora/bnc-100k.csv --features word lemma pos --max-dist 2

`--max-dist` tells how many different binary indexes that will be created: 
it's the maximum adjacent distance between the tokens in the query. The default setting is 2. 
If you only want unary indexes you can set `--max-dist` to 0, 
and if you want more control you can use the `--templates` option instead of `--features`.

The original `.csv` file is not used when searching, so you can remove it if you want 
(but then you cannot build any new indexes, so it's probably a bad idea).

### Building indexes faster

Note that it can take quite some time to build indexes for large corpora. 
But using [PyPy](https://www.pypy.org/) instead of CPython will be around twice as fast.

    pypy3 build_indexes.py --corpus ...

Most of the time spent when building indexes are for sorting them. 
There are several possible sorting implementations you can test with (using the `--sorter` option):

- `tmpfile` (the default) uses a temporary file which is sorted. 
  The main advantage is that it doesn't use up any internal memory, so it is useful for very large corpora.

- `internal` uses Python's builtin sort function, which is extremely fast
  (up to 5–10 times faster than `tmpfile`).
  However, it has to load the whole corpus in memory so it is not very useful for very large corpora
  (depending on your computer, but can cause problems from 10–100 million tokens).

- `cython` uses a Cython implementation using C's builtin `qsort` function for sorting the index.
  – this is also up to 5–10 times faster than `tmpfile`, but doesn't have any memory problems
  because it uses a temporary file.
  Note that you have to compile the Cython module first, by running `make faster-index-builder`.

## Searching from the command line

When you have to have built the inverted indexes as described above, you can search them.
To do this from the command line, you give the corpus and the query:

    python3 search_cmdline.py --corpus corpora/bnc-100k --query '[pos="ART"] [lemma="small"] [pos="SUBST"]'

All searches are cached (in the directory `cache/`), if you don't want to use the cached result, you can use `--no-cache`. 
Use `--print json` for JSON output, `--start 42` to print from match 42, and `--num 14` to print at most 14 matches. 
Use the following for more help: 

    python3 search_cmdline.py --help

Note that this can take a couple of seconds for some very general searches on large corpora. 
If you use the fast intersection (see below) some searches can be up to 10 times faster.

### Fast merging of search results

When searching the part that takes the most time is to merge two search results (e.g., calculating the intersection). 
The default implementation is in pure Python (in `merge.py`), but there is a faster version implemented in Cython. 
To use this you first have to install [Cython](https://cython.readthedocs.io/en/stable/src/quickstart/install.html). 
Then you can compile the `fast_merge` module:

    make fast-merge


## Using the web demo

To use the web demo you need to install [FastAPI](https://pypi.org/project/fastapi/). 
Before running the demo you also have to build search indexes for the corpora you want to play with. 
The you can run the following to start the webserver:

    uvicorn search_fastapi:app --reload

When the server has started you can open the URL <http://127.0.0.1:8000/webdemo/index.html> to play around with the web demo.

