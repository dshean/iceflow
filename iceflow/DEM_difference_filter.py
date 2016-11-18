#! /usr/bin/env python

# to run:
# 2 arguments:  ./srtm_proc.py '../data/AST14DEM/Nepal/AST14DEM_00310232003045929_20161116155805_13266.tif' 30
# 3 arguments:  ./srtm_proc.py '../data/AST14DEM/Nepal/AST14DEM_00310232003045929_20161116155805_13266.tif' '../data/SRTMv3/Nepal/SRTM_Nepal_mosaic.tif'  30

# use if permission denied: chmod u+x python_script.py



"""
David Shean modified by Iliyana Dobreva @geohackweek2016
dshean@gmail.com

Iterate over a directory and
Filter an input DEM using the dz_fltr function in filtlib (difference filter)
"""

import sys
import os

import glob


import numpy as np
from osgeo import gdal

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import filtlib

sys.path.insert(0, "..")

def dz_fltr_dir(dem_fn_list, refdem_fn, abs_dz_lim, out_dir):
    #names = dem_fn_list[0]
    for names in dem_fn_list:
        #print("Loading ouput DEM into masked array")
        dem_ds = iolib.fn_getds(names)
        dem_fltr = iolib.ds_getma(dem_ds, 1)

        #Difference filter, need to specify refdem_fn
        dem_fltr = filtlib.dz_fltr(names, refdem_fn, abs_dz_lim)

        # create output directory and file name
        parts = names.split('/')
        file_name = parts[len(parts)-1]

        dst_fn = out_dir +  file_name.split('.')[0] +'_filt%ipx.tif' % abs_dz_lim[1]

        print("Writing out filtered DEM: %s" % dst_fn)
        #Note: writeGTiff writes dem_fltr.filled()
        iolib.writeGTiff(dem_fltr, dst_fn, dem_ds)



def main():
    # change to accept input directory, SRTM, and tolerance
    # as arguments
    input_dir = '../data/AST14DEM/Nepal_2001/*.tif'
    out_dir = '../data/AST14DEM/Nepal_2001_filt60px/'

    refdem_fn = '../data/SRTMv3/Nepal/SRTM_Nepal_mosaic.tif'
    abs_dz_lim=(0,60) # tolerance

    # read file names in inut directory
    dem_fn_list = glob.glob(input_dir)

    dz_fltr_dir(dem_fn_list, refdem_fn, abs_dz_lim, out_dir)


# what does this do?
if __name__ == '__main__':
    main()
