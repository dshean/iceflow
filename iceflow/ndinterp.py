#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#This utility will interpolate to fill gaps in both spatial and temporal dimension

import os 
import sys

import multiprocessing as mp

from datetime import datetime, timedelta
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal
import scipy.interpolate
#from sklearn.gaussian_process import GaussianProcess
#from scikits import umfpack
from scipy.ndimage.interpolation import map_coordinates

from lib import iolib
from lib import malib
from lib import timelib 
from lib import geolib
from lib import pltlib

#This does the interpolation for a particular time for all points defined by x and y coords
#Used for parallel interpolation
def dto_interp(interpf, x, y, dto):
    return (dto, interpf(x, y, dto.repeat(x.size)).T)

def rangenorm(x, offset=None, scale=None):
    if offset is None:
        offset = x.min()
    if scale is None:
        scale = x.ptp()
    return (x.astype(np.float64) - offset)/scale

#This repeats the first and last array in the stack with a specified time offset
def pad_stack(s, dt_offset=timedelta(365.25)):
    o = s.ma_stack.shape
    new_ma_stack = np.ma.vstack((s.ma_stack[0:1], s.ma_stack, s.ma_stack[-1:]))
    new_date_list = np.ma.hstack((s.date_list[0:1] - dt_offset, s.date_list, s.date_list[-1:] + dt_offset))
    new_date_list_o = timelib.dt2o(new_date_list)
    return new_ma_stack, new_date_list_o

def apply_mask(a, m):
    a[:,m] = np.ma.masked

def main():
    
    if len(sys.argv) < 2: 
        sys.exit("Usage: %s stack.npz [mask.tif]" % os.path.basename(sys.argv[0]))
        
    #This will attempt to load cached files on disk
    load_existing = False 
    #Limit spatial interpolation to input mask
    clip_to_mask = True 

    #This expects a DEMStack object, see pygeotools/lib/malib.py or pygeotools/make_stack.py
    stack_fn = sys.argv[1]
    #Expects shp polygon as valid mask, in same projection as input raster
    mask_fn = sys.argv[2]

    stack = malib.DEMStack(stack_fn=stack_fn, save=False, trend=True, med=True, stats=True)
    #Get times of original obs
    t = stack.date_list_o.data
    t = t.astype(int)
    t[0] -= 0.1
    t[-1] += 0.1

    if clip_to_mask:
        m = geolib.shp2array(mask_fn, res=stack.res, extent=stack.extent)
        #Expand mask - hardcoded to 6 km
        import scipy.ndimage
        it = int(np.ceil(6000./stack.res))
        m = ~(scipy.ndimage.morphology.binary_dilation(~m, iterations=it))
        apply_mask(stack.ma_stack, m)

    #This is used frome here on out
    test = stack.ma_stack
    test_ptp = stack.dt_stack_ptp
    test_source = np.array(stack.source)
    res = stack.res
    gt = np.copy(stack.gt)

    #Probably don't need rull-res stack
    if True:
        stride = 2 
        test = test[:,::stride,::stride]
        test_ptp = test_ptp[::stride,::stride]
        res *= stride 
        print "Using a stride of %i (%0.1f m)" % (stride, res)
        gt[[1,5]] *= stride

    print "Orig shape: ", test.shape
    #Check to make sure all t have valid data
    tcount = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).count(axis=1)
    validt_idx = (tcount > 0).nonzero()[0]
    test = test[validt_idx]
    test_source = test_source[validt_idx]
    t = t[validt_idx]
    print "New shape: ", test.shape

    y, x = (test.count(axis=0) > 1).nonzero()
    x = x.astype(int)
    y = y.astype(int)
    #vm_t = test.reshape(test.shape[0], test.shape[1]*test.shape[2])
    vm_t = test[:,y,x]
    vm_t_flat = vm_t.ravel()
    idx = ~np.ma.getmaskarray(vm_t_flat)
    #These are values
    VM = vm_t_flat[idx]

    #Determine scaling factors for x and y coords
    #Should be the same for both 
    xy_scale = max(x.ptp(), y.ptp())
    xy_offset = min(x.min(), y.min())

    #This scales t to encourage interpolation along the time axis rather than spatial axis
    t_factor = 16. 
    t_scale = t.ptp()*t_factor
    t_offset = t.min()

    xn = rangenorm(x, xy_offset, xy_scale)
    yn = rangenorm(y, xy_offset, xy_scale)
    tn = rangenorm(t, t_offset, t_scale)

    X = np.tile(xn, t.size)[idx]
    Y = np.tile(yn, t.size)[idx]
    T = np.repeat(tn, x.size)[idx]
    #These are coords
    pts = np.vstack((X,Y,T)).T

    #Step size in days
    #ti_dt = 91.3125
    #ti_dt = 121.75 
    ti_dt = 365.25 

    #Set min and max times for interpolation
    #ti = np.arange(t.min(), t.max(), ti_dt)
    ti_min = timelib.dt2o(datetime(2008,1,1))
    ti_max = timelib.dt2o(datetime(2015,1,1))

    #Interpolate at these times 
    ti = np.arange(ti_min, ti_max, ti_dt)
    #Annual
    #ti = timelib.dt2o([datetime(2008,1,1), datetime(2009,1,1), datetime(2010,1,1), datetime(2011,1,1), datetime(2012,1,1), datetime(2013,1,1), datetime(2014,1,1), datetime(2015,1,1)])

    tin = rangenorm(ti, t_offset, t_scale)

    """
    #Never got this working efficiently, but preserved for reference
    #Radial basis function interpolation
    #Need to normalize to input cube  
    print "Running Rbf interpolation for %i points" % X.size
    rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='linear', smooth=0.1)
    #rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='gaussian', smooth=0.000001)
    #rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='inverse', smooth=0.00001)
    print "Sampling result at %i points" % xin.size
    vmi_rbf = rbfi(xin, yin, tin.repeat(x.size))
    vmi_rbf_ma[:,y,x] = np.ma.fix_invalid(vmi_rbf.reshape((ti.size, x.shape[0])))
    """

    #Attempt to load cached interpolation function 
    int_fn = '%s_LinearNDint_%i_%i.pck' % (os.path.splitext(stack_fn)[0], test.shape[1], test.shape[2]) 
    print int_fn
    if load_existing and os.path.exists(int_fn):
        print "Loading pickled interpolation function: %s" % int_fn
        f = open(int_fn, 'rb')
        linNDint = pickle.load(f)
    else:
        #NearestND interpolation (fast)
        #print "Running NearestND interpolation for %i points" % X.size
        #NearNDint = scipy.interpolate.NearestNDInterpolator(pts, VM, rescale=True)
        #LinearND interpolation
        print "Running LinearND interpolation for %i points" % X.size
        #Note: this breaks qhull for lots of input points
        linNDint = scipy.interpolate.LinearNDInterpolator(pts, VM, rescale=False)
        print "Saving pickled interpolation function: %s" % int_fn
        f = open(int_fn, 'wb')
        pickle.dump(linNDint, f, protocol=2)
        f.close()

    vmi_fn = '%s_%iday.npy' % (os.path.splitext(int_fn)[0], ti_dt)
    if load_existing and os.path.exists(vmi_fn):
        print 'Loading existing interpolated stack: %s' % vmi_fn
        vmi_ma = np.ma.fix_invalid(np.load(vmi_fn)['arr_0'])
    else:
        #Once tesselation is complete, sample each timestep in parallel
        print "Sampling %i points at %i timesteps, %i total" % (x.size, ti.size, x.size*ti.size)
        #Prepare array to hold output
        vmi_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
        """
        #This does all points at once
        #vmi = linNDint(ptsi)
        #vmi_ma[:,y,x] = np.ma.fix_invalid(vmi.reshape((ti.size, x.shape[0])))
        #This does interpolation serially by timestep
        for n, i in enumerate(ti):
            print n, i, timelib.o2dt(i)
            vmi_ma[n,y,x] = linNDint(x, y, i.repeat(x.size)).T
        """
        #Parallel processing
        pool = mp.Pool(processes=None)
        results = [pool.apply_async(dto_interp, args=(linNDint, xn, yn, i)) for i in tin]
        results = [p.get() for p in results]
        results.sort()
        for n, r in enumerate(results):
            t_rescale = r[0]*t_scale + t_offset
            print n, t_rescale, timelib.o2dt(t_rescale)
            vmi_ma[n,y,x] = r[1]

        vmi_ma = np.ma.fix_invalid(vmi_ma)
        print 'Saving interpolated stack: %s' % vmi_fn
        np.save(vmi_fn, vmi_ma.filled(np.nan))

    origt = False 
    if origt:
        print "Sampling %i points at %i original timesteps" % (x.size, t.size)
        vmi_ma_origt = np.ma.masked_all((t.size, test.shape[1], test.shape[2]))
        #Parallel
        pool = mp.Pool(processes=None)
        results = [pool.apply_async(dto_interp, args=(linNDint, x, y, i)) for i in t]
        results = [p.get() for p in results]
        results.sort()
        for n, r in enumerate(results):
            print n, r[0], timelib.o2dt(r[0])
            vmi_ma_origt[n,y,x] = r[1]
        vmi_ma_origt = np.ma.fix_invalid(vmi_ma_origt)
        #print 'Saving interpolated stack: %s' % vmi_fn
        #np.save(vmi_fn, vmi_ma.filled(np.nan))

    #Write out a proper stack, for use by stack_melt and flux gate mass budget
    if True:
        out_stack = deepcopy(stack)
        out_stack.stats = False
        out_stack.trend = False
        out_stack.datestack = False
        out_stack.write_stats = False
        out_stack.write_trend = False
        out_stack.write_datestack = False
        out_stack.ma_stack = vmi_ma
        out_stack.stack_fn = os.path.splitext(vmi_fn)[0]+'.npz'
        out_stack.date_list_o = np.ma.array(ti)
        out_stack.date_list = np.ma.array(timelib.o2dt(ti))
        out_fn_list = [timelib.print_dt(i)+'_LinearNDint.tif' for i in out_stack.date_list]
        out_stack.fn_list = out_fn_list
        out_stack.error = np.zeros_like(out_stack.date_list_o)
        out_stack.source = np.repeat('LinearNDint', ti.size)
        out_stack.gt = gt
        out_stack.res = res
        out_stack.savestack()

    sys.exit()


    """
    #Other interpolation methods
    #vmi = scipy.interpolate.griddata(pts, VM, ptsi, method='linear', rescale=True)

    #Kriging
    #Should explore this more - likely the best option
    #http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
    #http://resources.esri.com/help/9.3/arcgisengine/java/gp_toolref/geoprocessing_with_3d_analyst/using_kriging_in_3d_analyst.htm

    #PyKrige does moving window Kriging, but only in 2D
    #https://github.com/bsmurphy/PyKrige/pull/5

    #Could do tiled kriging with overlap in parallel
    #Split along x and y direction, preserve all t
    #Need to generate semivariogram globally though, then pass to each tile
    #See malib sliding_window
    wx = wy = 30
    wz = test.shape[0]
    overlap = 0.5
    dwx = dwy = int(overlap*wx)
    gp_slices = malib.nanfill(test, malib.sliding_window, ws=(wz,wy,wx), ss=(0,dwy,dwx))

    vmi_gp_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
    vmi_gp_mse_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))

    out = []
    for i in gp_slices:
        y, x = (i.count(axis=0) > 0).nonzero()
        x = x.astype(int)
        y = y.astype(int)
        vm_t = test[:,y,x]
        vm_t_flat = vm_t.ravel()
        idx = ~np.ma.getmaskarray(vm_t_flat)
        #These are values
        VM = vm_t_flat[idx]

        #These are coords
        X = np.tile(x, t.size)[idx]
        Y = np.tile(y, t.size)[idx]
        T = np.repeat(t, x.size)[idx]
        pts = np.vstack((X,Y,T)).T

        xi = np.tile(x, ti.size)
        yi = np.tile(y, ti.size)
        ptsi = np.array((xi, yi, ti.repeat(x.size))).T

        #gp = GaussianProcess(regr='linear', verbose=True, normalize=True, theta0=0.1, nugget=2)
        gp = GaussianProcess(regr='linear', verbose=True, normalize=True, nugget=2)
        gp.fit(pts, VM)
        vmi_gp, vmi_gp_mse = gp.predict(ptsi, eval_MSE=True)
        vmi_gp_ma = np.ma.masked_all((ti.size, i.shape[1], i.shape[2]))
        vmi_gp_ma[:,y,x] = np.array(vmi_gp.reshape((ti.size, x.shape[0])))
        vmi_gp_mse_ma = np.ma.masked_all((ti.size, i.shape[1], i.shape[2]))
        vmi_gp_mse_ma[:,y,x] = np.array(vmi_gp_mse.reshape((ti.size, x.shape[0])))
        out.append(vmi_gp_ma)
    #Now combine intelligently

    print "Gaussian Process regression"
    pts2d_vm = vm_t[1]
    pts2d = np.vstack((x,y))[~(np.ma.getmaskarray(pts2d_vm))].T
    pts2di = np.vstack((x,y)).T
    gp = GaussianProcess(regr='linear', verbose=True, normalize=True, theta0=0.1, nugget=1)
    gp.fit(pts, VM)
    print "Gaussian Process prediction"
    vmi_gp, vmi_gp_mse = gp.predict(ptsi, eval_MSE=True)
    print "Converting to stack"
    vmi_gp_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
    vmi_gp_ma[:,y,x] = np.array(vmi_gp.reshape((ti.size, x.shape[0])))
    vmi_gp_mse_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
    vmi_gp_mse_ma[:,y,x] = np.array(vmi_gp_mse.reshape((ti.size, x.shape[0])))
    sigma = np.sqrt(vmi_gp_mse_ma)
    """

    """
    #This fills nodata in last timestep with values from previous timestep
    #Helps Savitzy-Golay filter
    fill_idx = ~np.ma.getmaskarray(vmi_ma[-1]).nonzero()
    temp = np.ma.array(vmi_ma[-2])
    temp[fill_idx] = vmi_ma[-1][fill_idx]
    vmi_ma[-1] = temp
    """

if __name__ == "__main__":
    main()
