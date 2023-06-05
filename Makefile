
.PHONY: help clean java-arrays fast-intersection

help:
	@echo "'make java-arrays': build the Java implementation of Disk Fixed Size Arrays"
	@echo "'make fast-intersection': build the faster intersection module using Cython"
	@echo "'make clean': remove files built by the commands above"


clean:
	$(MAKE) -C java clean
	rm -f DiskFixedSizeArray.jar
	rm -f fast_intersection.c fast_intersection.cpython*.so


fast-intersection:
	python setup.py build_ext --inplace


java-arrays: DiskFixedSizeArray.jar

DiskFixedSizeArray.jar: java/*.java
	$(MAKE) -C java java-arrays
	cp java/$@ $@

