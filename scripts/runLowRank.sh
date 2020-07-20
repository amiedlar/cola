#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters
GLOBAL_STEPS=200
LOG_DIR=log/lowrank
OUT_DIR=out/lowrank
MAXRANK=${MAXRANK:-MAX_WORLD_SIZE}
#################### START: Experiments ####################
echo -e $"\e[1mSTART: Experiments\e[0m"
for (( RANK=1; RANK<=$MAXRANK; RANK++ ))
do
    NEW_DATASET=$DATASET'_rank'$RANK
    echo -e $"\e[2m"
    clean_dataset $NEW_DATASET true;
    colatools ${VERBOSE_FLAG} \
        load $DATASET \
        low-rank-approx $RANK \
        dump-svm --overwrite $NEW_DATASET \
        split --train $TRAIN_SIZE --seed $RANDOM_STATE $NEW_DATASET
    echo -e $"\e[0m"
    
    TOPOLOGY=complete
    run_cola_n $NEW_DATASET $MAX_WORLD_SIZE
    TOPOLOGY=ring
    run_cola_n $NEW_DATASET 3 $MAX_WORLD_SIZE
    clean_dataset $DATASET
    rm -rf $JOBLIB_CACHE_DIR
done