
.PHONY: help clean lint java-arrays fast-merge

help:
	@echo "'make java-arrays': build the Java implementation of Disk Fixed Size Arrays"
	@echo "'make fast-merge': build the faster merge module using Cython"
	@echo "'make clean': remove files built by the commands above"
	@echo "'make lint': type-check and linting using mypy, pyright, and ruff"


clean:
	$(MAKE) -C java clean
	rm -f DiskFixedSizeArray.jar
	rm -f fast_merge.c fast_merge.cpython*.so


PYVERSION = 3.9

# F541: Warn about f-strings without placeholders
RUFFCONFIG = 'lint.ignore = ["F541"]'

lint:
	mypy --python-version ${PYVERSION} --strict *.py || true
	@echo
	pyright --pythonversion ${PYVERSION} *.py || true
	@echo
	ruff check --config ${RUFFCONFIG} *.py || true


fast-merge:
	python3 setup.py build_ext --inplace


java-arrays: DiskFixedSizeArray.jar

DiskFixedSizeArray.jar: java/*.java
	$(MAKE) -C java java-arrays
	cp java/$@ $@

