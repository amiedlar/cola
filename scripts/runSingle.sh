#!/bin/bash
source scripts/colafunctions.sh
declare -g DATASET=${DATASET:-rderms}
declare -g K=${K:-5}
declare -g OVERSUB=${OVERSUB:-true}
declare -g TOPOLOGY=${TOPOLOGY:-complete}
set_cola_parameters

clean_dataset $DATASET true
echo -e $"\e[2m"
colatools load $DATASET \
    split --K $K --train $TRAIN_SIZE --seed $RANDOM_STATE $DATASET
echo -e $"\e[0m"

echo -e $"|-> Running CoLA, world size=$K, topology=$TOPOLOGY"
GLOBAL_STEPS=300
run_cola $DATASET $K

clean_dataset $DATASET