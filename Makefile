
DiskFixedSizeArray.jar: java/*.java
	$(MAKE) -C java
	cp java/$@ $@

