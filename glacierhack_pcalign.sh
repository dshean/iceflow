#! /bin/bash

#basic pc_align workflow

#define variables

max_displacement=1000

num_itt=2000

ref_dem='AST14DEM_00310232003045929_20161116155805_13266.tif'

source_dem='AST14DEM_00311102004045819_20161116155805_13272.tif'

output='AST14DEM_00311102004045819_20161116155805_13272_align'


echo 'create mask'

dem_mask.py $ref_dem



echo 'masked reference'

echo 'starting co-reg'

#coreg two dems
pc_align --max-displacement $max_displacement --num-iterations $num_itt \
--compute-translation-only --save-transformed-source-points \
$ref_dem $source_dem -o $output | tee test.log

echo 'co-reg complete. now converting point cloud to dem'

#create coreg rem
apply_dem_translation.py $source_dem test.log

echo 'done'