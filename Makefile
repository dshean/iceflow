#
# Makefile
# dlilien, 2016-11-16 14:20
#

all:
	git submodule update --init
	pip install -e demcoreg/
	pip install -e pygeotools/

# vim:ft=make
#
