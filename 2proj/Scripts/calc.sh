#!/usr/bin/env bash

IN_FILE=$1
SEQ_T=0

while read LINE
do
  N=`echo "${LINE}" |cut "-d;" -f 1`
  SIZE=`echo "${LINE}" |cut "-d;" -f 2`
  TIME_SCI=`echo "${LINE}" | cut "-d;" -f 10`
  T=`echo "${TIME_SCI}" | sed -e 's/[eE]+*/\\*10\\^/'`

  if [ $N -eq 1 ]
  then #line with sequential results
    echo $SIZE
    SEQ_T=$T
    #continue
  fi

  S=`echo "(${SEQ_T})/(${T})" | bc -l`
  E=`echo "(${S})/(${N})" | bc -l`
  ALFA=`echo "(${T} * ${N} - ${SEQ_T})/(${SEQ_T} * (${N} - 1))" | bc -l`
  echo -e "N = ${N}\tT(N) = ${T}\tS = ${S}\tE = ${E}\tALFA = ${ALFA}"
done < "${IN_FILE}"
echo
