#!/usr/bin/env bash

PROCS_LIST="1 2 4 8 16"
SIZES="16 32 64 128 256 512 1024 2048"
#SIZES="128 256 512 1024 2048"

for SIZE in $SIZES
do
  printf "${SIZE}: "

  #PROCS_LIST="$PROCS_LIST $SIZE"
  ../DataGenerator/arc_generator -N "${SIZE}" > /dev/null

  for PROCS in $PROCS_LIST
  do
    printf "${PROCS} "

    for I in `seq 10 10 101`
    do
      mpirun -np "${PROCS}" ./arc_proj02 -m 1 -w 1 -i arc_input_data.h5 -v -n "${I}" > /dev/null || exit 1
    done

    for I in `seq 1000 1000 10001`
    do
      mpirun -np "${PROCS}" ./arc_proj02 -m 1 -w 1 -i arc_input_data.h5 -v -n "${I}" > /dev/null || exit 1
    done

  done
  printf "\n"
done
