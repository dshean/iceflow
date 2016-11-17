#! /bin/bash

#basic pc_align workflow

if [ "$#" -ne 2 ] ; then
    echo "Usage is $0 ref_dem.tif src_dem.tif"
    echo "ref_dem.tif is the reference elevation data"
    echo "src_dem.tif is the DEM that will be adjusted to match ref_dem.tif"
    exit 1
fi

#Max displacement
max_displacement=1000

#Number of iterations
num_itt=2000

#ref_dem='AST14DEM_00310232003045929_20161116155805_13266.tif'
ref_dem=$1

#source_dem='AST14DEM_00311102004045819_20161116155805_13272.tif'
source_dem=$2

#pc_align output prefix
#output='AST14DEM_00311102004045819_20161116155805_13272_align'
output=${source_dem%.*}_align

#echo 'create mask'
#dem_mask.py $ref_dem
#echo 'masked reference'
#ref_dem=${ref_dem%.*}_masked.tif

#Define pc_align CLI arguments
pc_align_opt="--max-displacement $max_displacement"
pc_align_opt+=" --num-iterations $num_itt"
pc_align_opt+=" --compute-translation-only"
#This writes out an updated ASP pointcloud that can later be interpolated with point2dem
#We are going to shortcut using apply_dem_translation.py below
#pc_align_opt+=" --save-transformed-source-points"

echo; echo "Running pc_align"
log_fn=${output%.*}.log
pc_align $pc_align_opt $ref_dem $source_dem -o $output | tee $log_fn

echo; echo "Applying DEM translation"
apply_dem_translation.py $source_dem $log_fn 
