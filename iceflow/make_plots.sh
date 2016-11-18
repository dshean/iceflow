#! /bin/bash

#David Shean
#dshean@gmail.com

#Plotting commands for output iceflow products

fn_list=$(ls *DEM_32m.tif *DEM_32m_trans.tif)

#Generate shaded relief maps
hs.sh $fn_list

#Generate color shaded relief plots
parallel 'imview.py {} -overlay {.}_hs_az315.tif -scale x -label "Elevation (m WGS84)" -of png' ::: $fn_list

mos_fn=coreg32_mos
dem_mosaic -o $mos_fn $fn_list 
mos_fn+='-tile-0.tif'
hs.sh $mos_fn 

imview.py ${mos_fn}-tile-0 -overlay ${mos_fn}_hs_az315.tif -scale x -label 'Elevation (m WGS84)' -of png

make_stack.py $fn_list 
trend_fn=20100609_0450_1050410001B4E300_1050410001B4DC00-DEM_32m_20151018_0514_1050010001B64900_1050010001B64A00-DEM_32m_stack_4_trend.tif
#trend_fn=20100609_0450_1050410001B4E300_1050410001B4DC00-DEM_32m_trans_20151018_0514_1050010001B64900_1050010001B64A00-DEM_32m_stack_4_trend.tif

ndvtrim.py $trend_fn
trend_fn=${trend_fn%.*}_trim.tif

imview.py -cmap RdBu -clim -3 3 -label '2010-2015 dh/dt (m/yr)' $trend_fn -overlay ${mos_fn%.*}_hs_az315.tif -of png -scale x

rgi_fn=~/data/rgi50/regions/rgi50_merge.shp
~/src/iceflow/iceflow/raster_shpclip.py $trend_fn $rgi_fn 
trend_fn=${trend_fn%.*}_shpclip.tif
hs.sh $trend_fn

imview.py -cmap RdBu -clim -3 3 -label '2010-2015 dh/dt (m/yr)' $trend_fn -overlay ${mos_fn%.*}_hs_az315.tif -of png -scale x

