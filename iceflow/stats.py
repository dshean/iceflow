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

import warnings
warnings.filterwarnings('once', category=numpy.RankWarning)
def myfunc(vector):
    valid_mask = vector != 0
    try:
        return numpy.polyfit(timesteps[valid_mask], vector[valid_mask], 1)[0]
    except TypeError:
        return 0.

#for raster_tuples in itertools.izip(*(pygeoprocessing.iterblocks(r) for r in rasters)):
    # primitive workflow: use numpy.take to extract a single vector along the
    # dimension of matrices in raster_tuples.  This is problematic, though,
    # since we'll have to do this once for each pixel in the stack.
    # Ideally, we'll be able to do this on whole stacks of blocks at a time.
    # Approach: use apply_along_axis to calculate the regression this way.
    # TODO: Ignore nodata values so they don't skew results.

def mynewfunc_works(*blocks):
    applied = numpy.apply_along_axis(myfunc, 2, numpy.dstack(blocks))
    print applied.shape
    return applied

def mynewfunc(*blocks):
    stacked_array = numpy.dstack(blocks)
    applied = numpy.apply_along_axis(myfunc, 2, stacked_array)
    return applied

def mynewfunc_reshape(*blocks):
    stacked_array = numpy.dstack(blocks)
    print stacked_array.shape
    reshaped = numpy.reshape(stacked_array, -1)
    print reshaped.shape
    tiled = numpy.tile(timesteps, reshaped.shape[0])
    print tiled.shape
    tiled_timesteps = numpy.reshape(tiled, reshaped.shape[1])
    print tiled_timesteps
    valid_mask = reshaped == 0
    regression = numpy.polyfit(tiled_timesteps[valid_mask],
                               reshaped[valid_mask], 1)
    return regression.reshape(blocks[0].shape)

pygeoprocessing.vectorize_datasets(
    dataset_uri_list=rasters,
    dataset_pixel_op=mynewfunc,
    dataset_out_uri='newfile.tif',
    datatype_out=gdal.GDT_Float32,
    nodata_out=0,
    pixel_size_out=32.,
    bounding_box_mode='intersection',
    vectorize_op=False,
    datasets_are_pre_aligned=True)





