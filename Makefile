
.PHONY: help clean lint cython fast-merge qsort-index multikey-sort java-sort

help:
	@echo "'make clean': remove files built by the commands above"
	@echo "'make lint': type-check and linting using mypy, pyright, and ruff"
	@echo "'make cython': build all Cython modules"
	@echo "'make fast-merge': build the 'fast_merge' module for merging qery sets"
	@echo "'make qsort-index': build the 'qsort_index' module for search index sorting"
	@echo "'make multikey-sort': build the 'multikey_quicksort' module for search index sorting"
	@echo "'make java-sort': build the Java implementation of the search index sorter"


clean:
	$(MAKE) -C java clean
	rm -f *.jar
	rm -f *.c
	rm -f *.cpython*.so
	rm -rf build


PYVERSION = 3.9

# F541: Warn about f-strings without placeholders
RUFFCONFIG = 'lint.ignore = ["F541"]'

lint:
	mypy --python-version ${PYVERSION} --strict *.py || true
	@echo
	pyright --pythonversion ${PYVERSION} *.py || true
	@echo
	ruff check --config ${RUFFCONFIG} *.py || true


cython: fast-merge qsort-index multikey-sort

fast-merge: fast_merge.c
qsort-index: qsort_index.c
multikey-sort: multikey_quicksort.c

%.c: %.pyx
	cythonize -i $^


java-sort: DiskFixedSizeArray.jar

DiskFixedSizeArray.jar: java/*.java
	$(MAKE) -C java java-arrays
	cp java/$@ $@

