import os
import sys
import datetime
import logging
logging.basicConfig(level=logging.INFO)

import numpy
from osgeo import gdal
import pygeoprocessing


# dh/dt

# AST_YYYYMMDD_warp.TIF
def _date_from_filename(filename):
    time = os.path.basename(filename).split('_')[0]
    return datetime.date(int(time[0:4]),
                         int(time[4:6]),
                         int(time[6:]))

# Raster path, num days,
#rasters = [
#    '/data/AST_20160104_warp.TIF',
#    '/data/AST_20160205_warp.TIF',
#    '/data/AST_20160315_warp.TIF',
#]

import glob
rasters = sorted(glob.glob('data/*.tif'),
                 key=lambda x: int(os.path.basename(x).split('_')[0]))
print rasters

timesteps = [_date_from_filename(r) for r in rasters]
timesteps = numpy.array([(d - timesteps[0]).days for d in timesteps])

def regression(*blocks):
    stacked_array = numpy.dstack(blocks)
    new_shape = (stacked_array.shape[0]*stacked_array.shape[1], 4)
    reshaped = numpy.swapaxes(numpy.reshape(stacked_array, new_shape), 0, 1)
    regression = numpy.polyfit(timesteps,
                               reshaped, 1)[0]
    out_block = regression.reshape(blocks[0].shape)
    return numpy.where(numpy.min(stacked_array, axis=2) == 0, 0, out_block)


pygeoprocessing.vectorize_datasets(
    dataset_uri_list=rasters,
    dataset_pixel_op=regression,
    dataset_out_uri='newfile.tif',
    datatype_out=gdal.GDT_Float32,
    nodata_out=0,
    pixel_size_out=32.,
    bounding_box_mode='intersection',
    vectorize_op=False,
    datasets_are_pre_aligned=True)





