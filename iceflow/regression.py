import os
import glob
import sys
import datetime
import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('iceflow.regression')

import numpy
from osgeo import gdal
import pygeoprocessing



def _date_from_filename(filename):
    time = os.path.basename(filename).split('_')[0]
    return datetime.date(int(time[0:4]),
                         int(time[4:6]),
                         int(time[6:]))


def make_regression(worldview_folder, out_filename):
    rasters = sorted(glob.glob(worldview_folder + '/*.tif'),
                     key=lambda x: int(os.path.basename(x).split('_')[0]))
    LOGGER.info('Using rasters %s', rasters)

    timesteps = [_date_from_filename(r) for r in rasters]
    timesteps = numpy.array([(d - timesteps[0]).days for d in timesteps])
    LOGGER.info('Timesteps: %s' % timesteps)

    def _regression(*blocks):
        stacked_array = numpy.dstack(blocks)
        new_shape = (stacked_array.shape[0]*stacked_array.shape[1], len(timesteps))
        reshaped = numpy.swapaxes(numpy.reshape(stacked_array, new_shape), 0, 1)
        regression = numpy.polyfit(timesteps,
                                   reshaped, 1)[0]
        out_block = regression.reshape(blocks[0].shape)
        return numpy.where(numpy.min(stacked_array, axis=2) == 0, 0, out_block)

    pygeoprocessing.vectorize_datasets(
        dataset_uri_list=rasters,
        dataset_pixel_op=_regression,
        dataset_out_uri=out_filename,
        datatype_out=gdal.GDT_Float32,
        nodata_out=0,
        pixel_size_out=32.,
        bounding_box_mode='intersection',
        vectorize_op=False,
        datasets_are_pre_aligned=False)

if __name__ == '__main__':
    make_regression('data', 'regression.tif')
