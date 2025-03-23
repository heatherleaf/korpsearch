
.PHONY: help clean lint cython fast-merge qsort-index multikey-sort java-sort

help:
	@echo "'make clean': remove files built by the commands above"
	@echo "'make lint': type-check and linting using mypy, pyright, and ruff"
	@echo "'make cython': build all Cython modules"
	@echo "'make fast-merge': build the 'fast_merge' module for merging qery sets"
	@echo "'make faster-index-builder': build the 'faster_index_builder' module for index building"


clean:
	rm -f *.c
	rm -f *.*.so
	rm -rf build


PYVERSION = 3.10

# F541: Warn about f-strings without placeholders
RUFFCONFIG = 'lint.ignore = ["F541"]'

lint:
	mypy --python-version ${PYVERSION} --strict --no-warn-unused-ignores *.py || true
	@echo
	pyright --pythonversion ${PYVERSION} *.py || true
	@echo
	ruff check --config ${RUFFCONFIG} *.py || true


cython: fast-merge faster-index-builder

fast-merge: fast_merge.c
faster-index-builder: faster_index_builder.c

%.c: %.pyx
	cythonize -i $^
