#!/usr/bin/env python
#coding=utf-8

"""
Description: Compute glacier mass balance from a dhdt raster and a shapefile containing glaciers outlines.

Things to improve:
- keep only polygons that are fully in the raster extent
- add an option to read SLA in a shapefile
- save the results in a text file or in the shapefile

Author: Amaury Dehecq
Date: 17/11/2016
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np

# Personal libraries
from geoutils import geovector as vect
from geoutils.demraster import DEMRaster
import georaster as raster



def compute_mass_balance(dhfile,shapefile,dt,nodata=-9999,area_thresh=2,ice_density=0.85,SLA=None,snow_density=None,demfile=None,plot=False,backgrdfile=None):
    """
    Compute glacier mass balance from a dhdt raster and a shapefile containing glaciers outlines.
    Input:
    - dhfile: str, path to the dhdt raster file
    - shapefile: str, path to the shapefile containing glaciers outlines
    - dt: f, time span over which dh is computed
    - nodata: f, no data value in the dh file (Default is -9999)
    - area_thresh: f, if shapefile contains an Area attributes (like RGI), filter glaciers that have an area below this threshold
    - ice density: f, density of ice, used to convert meters to meters water equivalent (default is 0.85)
    - SLA = f, snow line altitude, if provided, will estimate the mass balance by mutliplying dh by ice_density below SLA and snow_density above SLA (default is None)
    - snow_density: f, snow/firn density, used to convert meters to meters water equivalent
    - demfile: str, path to the reference DEM (used to estimate points above/below SLA)
    - plot: bool, if True, display results
    - backgrdfile: str, path to a background file, will be displayed with a gray scale (Default is None)
    """

    print "Read input data"
    # Create goeraster instance
    dh_obj = raster.SingleBandRaster(dhfile)
    notvalid = np.where(dh_obj.r==nodata,True,False)

    # Create a geovector instance, cropped the dh extent
    outlines = vect.SingleLayerVector(shapefile)
    outlines.layer.SetAttributeFilter('AREA>%f' %area_thresh)
    outlines.crop2raster(dh_obj)
    outlines = outlines.reproject(dh_obj.srs)
    outlines.read()

    # Plot
    if plot==True:
        print "First plot"
        vmax = max(np.abs(np.percentile(dh_obj.r[dh_obj.r!=nodata],5)),np.abs(np.percentile(dh_obj.r[dh_obj.r!=nodata],95)))
        plt.imshow(np.ma.array(dh_obj.r,mask=np.where(dh_obj.r==nodata,True,False)),vmin=-vmax,vmax=vmax,cmap=plt.get_cmap('RdBu'),interpolation='nearest',extent=dh_obj.extent)
        outlines.draw(facecolor='none')
        cb = plt.colorbar()
        cb.set_label('dh (m)')
        plt.show()
    
    

    # Convert meters to meters water equivalent
    if SLA==None:
        dh_obj.r = dh_obj.r * ice_density
        
    else:
        # Read reference DEM and reproject to raster projection
        dem = DEMRaster(demfile,load_data=False)
        dem_proj = dem.reproject(dh_obj.srs, dh_obj.nx, dh_obj.ny, dh_obj.extent[0], dh_obj.extent[3], dh_obj.xres, dh_obj.yres, nodata=dem.ds.GetRasterBand(1).GetNoDataValue())
        dem_proj = DEMRaster(dem_proj.ds)   # reproject will return a georaster.SingleBandRaster instance, convert it to a DEMRaster instance

        # convert to meters water equivalent
        dh_obj.r = np.where(dem_proj.r<SLA, dh_obj.r * ice_density, dh_obj.r * snow_density)


    # Apply back no data value
    dh_obj.r[notvalid] = nodata
        
    # Compute mean dh for each polygon in the dh raster extent
    dh_mean, dh_std, dh_count, dh_frac =outlines.zonal_statistics(dh_obj,nodata=nodata)

    # Compute dhdt
    dhdt = dh_mean/dt

    # Print results
    # Could add a function to save in a text file or save in the shapefile
    print "\n#############################"
    print "Glacier RGI id, mass balance (m w.e/yr)"
    for k in xrange(outlines.FeatureCount()):
        print outlines.fields.values['RGIId'][k], dhdt[k]
    print "#############################\n"

    
    # Plotting
    if plot==True:
        print "Second plot"
        # Save in shapefile attributes for plotting
        outlines.fields.values['dhdt'] = dhdt

        # Read backrground image
        if backgrdfile!=None:
            backgrd = DEMRaster(backgrdfile,load_data=False)
            backgrd_proj = backgrd.reproject(dh_obj.srs, dh_obj.nx, dh_obj.ny, dh_obj.extent[0], dh_obj.extent[3], dh_obj.xres, dh_obj.yres, nodata=backgrd.ds.GetRasterBand(1).GetNoDataValue())
        
        vmax = max(np.abs(np.nanmin(dhdt)),np.abs(np.nanmax(dhdt)))
        
        if backgrdfile!=None:
            plt.imshow(backgrd_proj.r,cmap='Greys',extent=dh_obj.extent)
            
        outlines.draw_by_attr('dhdt',vmin=-vmax,vmax=vmax,cmap=plt.get_cmap('RdBu'),clabel='dhdt (m.w.e./yr)')
        plt.xlim(dh_obj.extent[0],dh_obj.extent[1])
        plt.ylim(dh_obj.extent[2],dh_obj.extent[3])
        plt.show()


if __name__=='__main__':

    import argparse

    #Set up arguments
    parser = argparse.ArgumentParser(description='Compute glacier mass balance from a dhdt raster and a shapefile containing glaciers outlines.')


    #Positional arguments
    parser.add_argument('dhfile', type=str, help='str, path to the dhdt raster file')
    parser.add_argument('shapefile', type=str, help='str, path to the shapefile containing glaciers outlines')
    parser.add_argument('dt', type=float, help='f, time span over which dh is computed')

    
    #optional arguments
    parser.add_argument('-nodata', dest='nodata', type=float, default=-9999, help='f, no data value in the dh file (Default is -9999)')
    parser.add_argument('-area_thresh', dest='area_thresh', type=float, default=2, help='f, if shapefile contains an Area attributes (like RGI), filter glaciers that have an area below this threshold')
    parser.add_argument('-ice_density', dest='ice_density', type=float, default=0.85, help='f, density of ice, used to convert meters to meters water equivalent (default is 0.85)')
    parser.add_argument('-SLA', dest='SLA', type=float, default=None, help='f, snow line altitude, if provided, will estimate the mass balance by mutliplying dh by ice_density below SLA and snow_density above SLA (default is None)')
    parser.add_argument('-snow_density', dest='snow_density', type=float, default=0.6, help='f, snow/firn density, used to convert meters to meters water equivalent')
    parser.add_argument('-demfile', dest='demfile', type=str, default=None, help='demfile: str, path to the reference DEM (used to estimate points above/below SLA)')
    parser.add_argument('-plot', dest='plot', help='if True, display results (Default is False)',action='store_true')
    parser.add_argument('-backgrdfile', dest='backgrdfile', type=str, default=None, help="str, path to a background file, will be displayed with a gray scale (Default is None)")

    args = parser.parse_args()

    # Run main function
    compute_mass_balance(args.dhfile,args.shapefile,args.dt,args.nodata,args.area_thresh,args.ice_density,args.SLA,args.snow_density,args.demfile,args.plot,args.backgrdfile)
    
    # dhfile = '/disk/L0data/DhDtGardelle/orig/Everest_dh_ON_mask.tif'
    # shapefile = '/disk/L0data/shapefiles/RGI_5.0/all_asia.shp'
    # demfile = '/disk/L0data/DEM/SRTM/srtm-himalaya-1arcsec_shaded.dem.TIF'
    # backgrdfile = '/disk/L0data/DEM/SRTM/srtm-himalaya-1arcsec_shaded.dem.TIF'
    # compute_mass_balance(dhfile,shapefile,dt=11,plot=False,backgrdfile=backgrdfile)

    


