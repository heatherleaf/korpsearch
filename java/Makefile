
.PHONY: help clean java-arrays fast-intersection

help:
	@echo "'make java-arrays': build the Java implementation of Disk Fixed Size Arrays"
	@echo "'make clean': remove files built by the command above"


clean:
	rm -f DiskFixedSizeArray.jar *.class


java-arrays: DiskFixedSizeArray.jar

DiskFixedSizeArray.jar: DiskFixedSizeArray.java Quicksort.java
	javac $^
	jar cvfe $@ $(basename $<) $(addsuffix *.class,$(basename $^))

