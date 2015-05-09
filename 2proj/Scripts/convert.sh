#!/usr/bin/env bash

cd $1

for LINE in {2..11}
do
  for FILE in out*.csv
  do
    REC=$(head -$LINE $FILE | tail -1)
    CPUS=${FILE#out_}
    CPUS="${CPUS%.csv}"
    if [ $CPUS == "seq" ]
    then
      CPUS=1
    fi
    echo "$CPUS;$REC"
  done > $LINE.txt
done
