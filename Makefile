
.PHONY: help clean lint cython fast-merge qsort-index multikey-sort java-sort

help:
	@echo "'make clean': remove files built by the commands above"
	@echo "'make lint': type-check and linting using mypy, pyright, and ruff"
	@echo "'make cython': build all Cython modules"
	@echo "'make faster-index-builder': build the 'faster_index_builder' module for index building"


clean:
	rm -f *.c
	rm -f *.*.so
	rm -rf build


PYVERSION = 3.10

# F541: Warn about f-strings without placeholders
# E402: Module level import not at top of file
# E731: Do not assign a `lambda` expression, use a `def`
RUFFCONFIG = 'lint.ignore = ["F541", "E402", "E731"]'
MYPYCONFIG = --strict --no-warn-unused-ignores

lint:
	mypy --python-version ${PYVERSION} ${MYPYCONFIG} *.py || true
	@echo
	pyright --pythonversion ${PYVERSION} *.py || true
	@echo
	ruff check --config ${RUFFCONFIG} *.py || true


cython: faster-index-builder

faster-index-builder: faster_index_builder.c

%.c: %.pyx
	cythonize -i $^
