#! /bin/bash

#Wrapper for Ames Stereo Pipeline (ASP) pc_align command line utility
#Used to co-register two input point clouds or DEMs

#Requires that ASP has been installed and executables are in PATH
#Also requires pygeotools apply_dem_translation.py is executable and in PATH

#See ASP website for precompiled binaries and official documentation: 
#https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/

#Parse command line arguments
if [ "$#" -ne 2 ] ; then
    echo "Usage is $0 ref_dem.tif src_dem.tif"
    echo "ref_dem.tif is the reference elevation data"
    echo "src_dem.tif is the DEM that will be adjusted to match ref_dem.tif"
    exit 1
fi

#Max displacement sets max inital error. helps to reduce wasted itterations
max_displacement=1000

#max number of iterations for pc_align to use.
num_itt=2000

#attach command line arguments to variables for use.
ref_dem=$1
source_dem=$2

#pc_align output prefix
output=${source_dem%.*}_align

# generate bare earth mask for co-registration
# not implemented due to time constraints of geohack
#dem_mask.py $ref_dem
#echo 'masked reference'
#ref_dem=${ref_dem%.*}_masked.tif

#Define pc_align CLI arguments
pc_align_opt="--max-displacement $max_displacement"
pc_align_opt+=" --num-iterations $num_itt"
pc_align_opt+=" --compute-translation-only"

# runs pc align and generates log file with translation parameters
echo; echo "Running pc_align"
log_fn=${output%.*}.log
pc_align $pc_align_opt $ref_dem $source_dem -o $output | tee $log_fn

#Apply the horizontal translation to input DEM geotransform, remove vertical offset
echo; echo "Applying DEM translation"
apply_dem_translation.py $source_dem $log_fn 
