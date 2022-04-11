
# Korp-search

Searching in very large corpora.

There are currently two variants of the search database:

- using the `shelve` module to store dictionaries of sets of sentences
- using a file-based hash table for the same thing

## Building indexes

The following builds search indexes for the BNC-mini corpus using the features `word`, `lemma`, and `pos`:

```
python korpsearch.py corpora/bnc-mini.csv --build-index --features word lemma pos
```

There are four different variants of building and storing the indexes:

- `--algorithm binsearch` (the default): Hashes the queries into numbers, and uses binary search to look them up
- `--algorithm hashtable`: Hashes the queries into numbers, and stores them as an open addressing hash table (without probing)
- `--algorithm instance`: Does not hash the queries but stores them directly as strings, and uses binary search (this uses more disk space)
- `--algorithm shelve`: Uses the `shelve` module to store the database on disk (this uses considerably more disk space)

## Search results

Searching in a corpus results in an `IndexSet`, which is a disk-based set representation that has one main operation: set intersection.

After intersecting an IndexSet with another set, the result is stored as a standard Python set (which uses internal memory).

## Searching

To search, you just provide a query as the second argument:

```
python korpsearch.py corpora/bnc-mini.csv '[pos="ART"] [lemma="small"] [pos="SUBST"]' 
```
