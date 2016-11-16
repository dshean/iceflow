#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Little utility for volume change analysis from dz rasters
#Input should already be clipped appropriately (e.g. masked except over glaciers)

import sys
import os
import datetime

import numpy as np

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import geolib
from pygeotools.lib import timelib

def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: %s dz.tif" % os.path.basename(sys.argv[0]))

    #This is mean density for N Cascades snow
    #rho = 0.5
    #Density of pure ice 
    rho = 0.917

    #Clip negative values to 0
    filt = False 

    src_fn = sys.argv[1]
    src_ds = iolib.fn_getds(src_fn)
    res = geolib.get_res(src_ds, square=True)[0]
    bma = iolib.ds_getma(src_ds)

    #Attempt to extract t1 and t2 from input filename
    ts = timelib.fn_getdatetime_list(src_fn)
    #Hardcode timestamps
    #ts = [datetime.datetime(2013,9,10), datetime.datetime(2014,5,14)]

    dt_yr = None
    if len(ts) == 2:
        dt = ts[1] - ts[0]
        year = datetime.timedelta(days=365.25)
        dt_yr = dt.total_seconds()/year.total_seconds()

    #Can add filter here to remove outliers, perc_fltr(0.01, 99.9)
    if filt:
        mask = np.ma.getmaskarray(bma)
        bma[bma < 0] = 0
        bma = np.ma.array(bma, mask=mask)

    #Print out stats
    print('\n')
    stats = malib.print_stats(bma)

    count = stats[0]
    area = res**2*count
    mean = stats[3]
    med = stats[5]

    s_m3 = np.ma.sum(bma)*res**2 
    s_km3 = s_m3/1E9 
    s_mwe = mean*rho
    s_gt = s_km3*rho
    s_mm = s_gt/374

    if dt_yr is not None:
        print("%s to %s: %0.2f yr" % (ts[0], ts[1], dt_yr))
        print("%0.0f m^3 (%0.0f m^3/yr)" % (s_m3, s_m3/dt_yr))
        print("%0.3f km^3 (%0.3f km^3/yr)" % (s_km3, s_km3/dt_yr))
        print("Density: %0.3f g/cc" % rho)
        print("%0.3f GT (%0.3f GT/yr)" % (s_gt, s_gt/dt_yr))
        print("%0.6f mm SLR (%0.6f mm/yr)" % (s_mm, s_mm/dt_yr))
        print("%0.3f m.w.e. (%0.3f m.w.e./yr)" % (s_mwe, s_mwe/dt_yr))
    else:
        print("Area: %0.2f km2" % (area/1E6))
        print("%0.0f m^3" % s_m3)
        print("%0.3f km^3" % s_km3) 
        print("Density: %0.3f g/cc" % rho)
        print("%0.3f GT" % s_gt)
        print("%0.6f mm SLR" % s_mm)
        print("%0.3f m.w.e." % s_mwe)

if __name__ == "__main__":
    main()
