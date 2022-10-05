
DiskFixedSizeArray.jar: DiskFixedSizeArray.java Quicksort.java
	javac $^
	jar cvfe $@ $(basename $<) $(addsuffix *.class,$(basename $^))

