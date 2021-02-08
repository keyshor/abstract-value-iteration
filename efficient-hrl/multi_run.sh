#!/bin/bash

DIRECTORY=$1
if [[ $1 == "ant_maze" ]]; then
    GIN_FILE=$1
else
    GIN_FILE="${1}_single"
fi
UVF=base_uvf
PORT=$2

for i in {3..9}
do
    tensorboard --logdir ${DIRECTORY}/eval${i} --port $(( $PORT + $i ))  & python scripts/local_eval.py ${i} hiro_orig ${GIN_FILE} ${UVF} ${DIRECTORY} & python scripts/local_train.py ${i} hiro_orig ${GIN_FILE} ${UVF} ${DIRECTORY} &
done