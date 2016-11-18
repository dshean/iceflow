#! /usr/bin/env python

# dshean@gmail.com
# 4/8/13

# This script will downsample an input raster with gaussian pyramid approach and then fill any remaining holes
# Current approach uses the GDAL inpainting method
# Potentially incorporate OpenCV if inpainting supports Float32

# To do:
# Write out unfilled tif
# Experiment with smoothing during filling and max distance
# Write gdal-dev about valid/invalid issue and border ndv - need three values (0, 1, 2)
# Implement loop to do both 4x and 16x

import argparse
import gdal
import numpy as np
from pygeotools.lib import malib, iolib
import os

memdrv = gdal.GetDriverByName('MEM')
opt = iolib.gdal_opt


def gdalfill_ds(ds, prog_func=None):
    # Create temp ds in memory to avoid writing in place (default
    # gdal.FillNodata behavior)
    tmp_ds = memdrv.CreateCopy('', ds, 1)
    for n in range(1, ds.RasterCount + 1):
        tmp_b = tmp_ds.GetRasterBand(n)
        b_fill = gdalfill_b(tmp_b, prog_func=prog_func)
        tmp_b.WriteArray(b_fill.ReadAsArray())
    return tmp_ds


def gdalfill_b(b, edgemask=True, prog_func=None):
    # Create mask of exterior nodata
    bma = iolib.b_getma(b)

    # Check to make sure there are actually holes
    # if bma.count_masked() > 0:
    if np.any(bma.mask):
        # Create 8-bit mask_ds, and add edgemask
        mask_ds = memdrv.Create('', b.XSize, b.YSize, 1, gdal.GDT_Byte)
        maskband = mask_ds.GetRasterBand(1)
        # The - here inverts the mask so that holes are set to 0 (invalid) and
        # filled by gdalFillNodata
        maskband.WriteArray((-bma.mask).astype(int))

        # Now fill holes in the output
        print("Filling holes")
        max_distance = 40
        smoothing_iterations = 0
        fill_opt = []
        gdal.FillNodata(b, maskband, max_distance,
                        smoothing_iterations, fill_opt, callback=prog_func)

        # Apply the original edgemask
        # Note: need to implement convexhull option like geolib.get_outline
        if edgemask:
            print("Generating outer edgemask")
            edgemask = malib.get_edgemask(bma)
            out_ma = np.ma.array(b.ReadAsArray(), mask=edgemask)
        else:
            out_ma = np.ma.array(b.ReadAsArray())
        # Set ndv to input ndv
        out_ma.set_fill_value(bma.fill_value)

        # Write the filled, masked array to our tmp dataset
        b.WriteArray(out_ma.filled())

        # Free mask_ds
        mask_ds = None
    else:
        print("No holes found in input band")
    return b


def writeout(dst_fn, ds):
    print("Writing out", dst_fn)
    gtifdrv = gdal.GetDriverByName('GTiff')
    dst_ds = gtifdrv.CreateCopy(dst_fn, ds, 1, options=opt)
    #dst_ds = driver.Create(dst_fn, ovr_b.XSize, ovr_b.YSize, 1, ovr_b.DataType, options=opt)
    del dst_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', help='Filename of raster to inpaint')
    parser.add_argument('-o', default=None, type=str, help='Output filename. Default is in + _fill.tif')
    args = parser.parse_args()

    prog_func = gdal.TermProgress
    # The ReadOnly forces overviews to be written to external file
    src_ds = gdal.Open(args.fn, gdal.GA_ReadOnly)

    if args.o is None:
        fill_fn = os.path.splitext(args.fn)[0] + '_fill.tif'
    else:
        fill_fn = args.o
    print("Filling %s" % args.fn)
    fill_ds = gdalfill_ds(src_ds, prog_func)
    writeout(fill_fn, fill_ds)

    src_ds = None

if __name__ == "__main__":
    main()
