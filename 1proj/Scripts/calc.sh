#!/usr/bin/env bash

IN_FILE=$1
OLD_SIZE=0
SEQ_T=0
OLD_SIM_MODE="none"

while read LINE
do
  NEW_SIZE=`echo "${LINE}" | cut "-d;" -f 1`
  N=`echo "${LINE}" |cut "-d;" -f 3`
  NEW_SIM_MODE=`echo "${LINE}" | cut "-d;" -f 8`
  TIME_SCI=`echo "${LINE}" |cut "-d;" -f 10`
  T=`echo "${TIME_SCI}" | sed -e 's/[eE]+*/\\*10\\^/'`

  if [[ ! $NEW_SIZE =~ ^[0-9]+$ ]]
  then #size is not integer
    continue
  fi

  if [ $OLD_SIZE -ne $NEW_SIZE ]
  then #new size begins, line with sequential results
    echo -e "----- SIZE = ${NEW_SIZE}\tT(1) = ${T} -----"
    OLD_SIZE=$NEW_SIZE
    SEQ_T=$T
    continue
  fi

  if [ $OLD_SIM_MODE != $NEW_SIM_MODE ]
  then #switch between par1 and par2
    echo "SIMULATION MODE = ${NEW_SIM_MODE}"
    OLD_SIM_MODE=$NEW_SIM_MODE
  fi

  S=`echo "(${SEQ_T})/(${T})" | bc -l`
  E=`echo "(${S})/(${N})" | bc -l`
  ALFA=`echo "(${T} * ${N} - ${SEQ_T})/(${SEQ_T} * (${N} - 1))" | bc -l`
  echo -e "N = ${N}\tT(N) = ${T}\tS = ${S}\tE = ${E}\tALFA = ${ALFA}"
done < "${IN_FILE}"
