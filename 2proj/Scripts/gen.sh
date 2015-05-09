#!/usr/bin/env bash

module load intel/15.2.164

# domain sizes
declare -a sizes=(256 512 1024 2048 4096)

# generate input files
for size in ${sizes[*]} 
do
	echo "Generating input data ${size}x${size}..."
  ../DataGenerator/arc_generator -o input_data_${size}.h5 -N ${size} -H 100 -C 20
done
