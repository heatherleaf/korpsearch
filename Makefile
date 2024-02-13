
.PHONY: help clean java-arrays fast-merge

help:
	@echo "'make java-arrays': build the Java implementation of Disk Fixed Size Arrays"
	@echo "'make fast-merge': build the faster merge module using Cython"
	@echo "'make clean': remove files built by the commands above"


clean:
	$(MAKE) -C java clean
	rm -f DiskFixedSizeArray.jar
	rm -f fast_merge.c fast_merge.cpython*.so


fast-merge:
	python3 setup.py build_ext --inplace


java-arrays: DiskFixedSizeArray.jar

DiskFixedSizeArray.jar: java/*.java
	$(MAKE) -C java java-arrays
	cp java/$@ $@

