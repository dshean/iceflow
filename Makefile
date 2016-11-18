#
# Makefile
# dlilien, 2016-11-16 14:20
#

all:
	pip install -e .
	git submodule update --init
	pip install -e demcoreg/
	pip install -e pygeotools/
	pip install -e geoutils/

# vim:ft=make
#
