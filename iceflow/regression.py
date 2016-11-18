# TODO: Add a page on what we did to the wiki.
# TODO: Add order, weights to polyfit


import os
import glob
import datetime
import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('iceflow.regression')

import numpy
from osgeo import gdal
import pygeoprocessing

def _date_from_filename(filename):
    """Read in a datetime object from a Worldview raster filename.

    Filenames are expected to have the format:

        ``YYYYMMDD_HHMM_<otherstuff>.tif``

    The first eight characters are parsed into an instance of
    ``datetime.date`` representing this date.

    Parameters:
        filename (string): The string filename.

    Returns:
        An instance of ``datetime.date`` for the datestamp."""
    time = os.path.basename(filename).split('_')[0]
    return datetime.date(int(time[0:4]),
                         int(time[4:6]),
                         int(time[6:]))


def _compare_with_make_stack(stack_trend_file, pgp_trend_file, diff_file):
    """Compare trend of ``make_regression`` with trend from ``make_stack.py``.

    Comparison is done as a per-pixel diff on any pixel pairs where both
    pixels are not nodata.  If either pixel in a pixel stack is nodata, the
    stack is ignored and nodata is returned for that pixel value.

    The diff looks like this::

        diff_file = stack_trend_file - pgp_trend_file

    Parameters:
        stack_trend_file (string): The path to the trend raster output of
            ``make_stack.py`` (usually named ``stack_trend.tif``).  This
            file must exist on disk.
        pgp_trend_file (string): The path to the trend raster output from
            ``make_regression()``, also in this module.  This file must
            exist on disk.
        diff_file (string): The path to where the difference raster should be
            saved.

    Returns:
        ``None``"""
    stack_nodata = pygeoprocessing.get_nodata_from_uri(stack_trend_file)
    pgp_nodata = pygeoprocessing.get_nodata_from_uri(pgp_trend_file)

    def _diff(stack_trend, pgp_trend):
        """Calculate a diff between two matrices, ignoring nodata.

        Parameters:
            stack_trend (numpy.ndarray): Array of values from the stack trend
                raster.
            pgp_trend (numpy.ndarray): Array of values from the pygeoprocessing
                trend raster.

        Returns:
            ``numpy.ndarray`` of the difference between ``stack_trend`` and
            ``pgp_trend``"""
        valid_mask = ((stack_trend != stack_nodata) & (pgp_trend != pgp_nodata))
        out_array = numpy.empty_like(stack_trend)
        out_array[:] = -9999
        out_array[valid_mask] = stack_trend[valid_mask] - pgp_trend[valid_mask]
        return out_array

    pygeoprocessing.vectorize_datasets(
        dataset_uri_list=[stack_trend_file, pgp_trend_file],
        dataset_pixel_op=_diff,
        dataset_out_uri=diff_file,
        datatype_out=gdal.GDT_Float32,
        nodata_out=-9999,
        pixel_size_out=32.,
        bounding_box_mode='intersection',
        vectorize_op=False,
        datasets_are_pre_aligned=False)


def make_regression(worldview_folder, out_filename, deg=1, weights=None):
    """Calculate a regression between worldview DEMs within a folder.

    Note:
        Any pixel stacks that contain 1 or more nodata values will be
        excluded from the regression calculations.

        Additionally, this function assumes that all worldview rasters
        have a nodata value of ``0``.

    Parameters:
        worldview_folder (string): The path to a folder on disk containing
            GeoTiffs representing elevation data.  Any files with a ``'.tif``
            extension will be analyzed within this folder.  There is no
            upper limit to the number of files that can be analyzed.
        out_filename (string): The path on disk to where the regression raster
            should be stored.  If the file already exists on disk, it will be
            overwritten.
        deg=1 (int): The order of the regression.  Passed directly to
            ``numpy.polyfit`` via the ``deg`` parameter.  1 represents
            linear regression.
        weights=None (``numpy.ndarray`` or ``None``): If None, the inputs will
            be unweighted in the regression.  If an ``ndarray``, this array
            must a 1D array with the same length as there are files in
            ``worldview_folder``.

    Raises:
        ``ValueError``: When the length of ``weights`` does not equal
            the number of geotiffs found in ``worldview_folder``.

    Returns:
        ``None``"""
    rasters = sorted(glob.glob(worldview_folder + '/*.tif'),
                     key=lambda x: int(os.path.basename(x).split('_')[0]))
    LOGGER.info('Using rasters %s', rasters)

    timesteps = [_date_from_filename(r) for r in rasters]
    timesteps = numpy.array([(d - timesteps[0]).days for d in timesteps])
    LOGGER.info('Timesteps: %s' % timesteps)

    if weights and not (len(weights) == len(timesteps)):
        raise ValueError(('Weights length (%s) does not match timesteps '
                          'length (%s)') % (len(weights), len(timesteps)))

    def _regression(*blocks):
        """Compute linear regression from a stack of DEM matrices.

        Note:
            Any pixel stacks that contain 1 or more values of 0 will have an
            output value of ``0``.

        Parameters:
            blocks (list): A list of 2D ``numpy.ndarray`` instances with pixel
                values from the stack of rasters passed to
                ``vectorize_datasets`` call.  There is no upper limit to the
                number of timesteps that can be calculated.

        Returns:
            ``numpy.ndarray``, in 2 dimensions.  This will contain the ``m``
            parameter from the fitted line.
        """
        stacked_array = numpy.dstack(blocks)
        new_shape = (stacked_array.shape[0]*stacked_array.shape[1],
                     len(timesteps))
        reshaped = numpy.swapaxes(numpy.reshape(stacked_array, new_shape), 0, 1)
        regression = numpy.polyfit(timesteps,
                                   reshaped, deg=deg, w=weights)[0]
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
