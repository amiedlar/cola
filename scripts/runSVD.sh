#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters
TRAIN_SIZE=${TRAIN_SIZE:-0.7}
LINREG_ITER=$LINREG_ITER' --svd'
GLOBAL_STEPS=200
#################### START: Experiments ####################
echo -e $"\e[1mSTART: Experiments\e[0m"
echo -e $"\e[2m"
if [[ $REDUCED = true ]]; then
    MAX_WORLD_SIZE=3
    if [[ $SCALE = true ]]; then
    NEW_DATASET=$DATASET'_rsvd_scale'
    clean_dataset $NEW_DATASET true;
    colatools ${VERBOSE_FLAG} \
        load $DATASET \
        decompose --scale --threshold 1e0 svd \
        info-rank \
        info-cond --p 'fro' \
        info-cond --p 1 \
        info-cond --p 2 \
        dump-svm --overwrite $NEW_DATASET \
        split --train $TRAIN_SIZE --seed $RANDOM_STATE $NEW_DATASET
    DATASET=$NEW_DATASET
    else
    NEW_DATASET=$DATASET'_rsvd'
    clean_dataset $NEW_DATASET true;
    colatools ${VERBOSE_FLAG} \
        load $DATASET \
        decompose --threshold 1e0 svd \
        info-rank \
        info-cond \
        dump-svm --overwrite $NEW_DATASET \
        split --train $TRAIN_SIZE --seed $RANDOM_STATE $NEW_DATASET
    DATASET=$NEW_DATASET
    fi
else
    if [[ $SCALE = true ]]; then
    NEW_DATASET=$DATASET'_svd_scale'
    clean_dataset $NEW_DATASET true;
    colatools ${VERBOSE_FLAG} \
        load $DATASET \
        decompose --scale svd \
        info-rank \
        info-cond --p 'fro' \
        info-cond --p 1 \
        info-cond --p 2 \
        dump-svm --overwrite $NEW_DATASET \
        split --train $TRAIN_SIZE --seed $RANDOM_STATE $NEW_DATASET
    DATASET=$NEW_DATASET
    else
    NEW_DATASET=$DATASET'_svd'
    clean_dataset $NEW_DATASET true;
    colatools ${VERBOSE_FLAG} \
        load $DATASET \
        decompose svd \
        info-rank \
        info-cond \
        dump-svm --overwrite $NEW_DATASET \
        split --train $TRAIN_SIZE --seed $RANDOM_STATE $NEW_DATASET
    DATASET=$NEW_DATASET
    fi
fi
echo -e $"\e[0m"
###################### START: Complete #####################
echo -e $"\e[1mSTART: Complete\e[0m"

TOPOLOGY='complete' 
# Run CoLA 
run_cola_n $DATASET $MAX_WORLD_SIZE

echo -e $"\e[1mEND: Complete\e[0m\n"
####################### END: Complete ######################

######################## START: Ring #######################
echo -e $"\e[1mSTART: Ring\e[0m"

TOPOLOGY='ring' 
# Run CoLA 
run_cola_n $DATASET 3 $MAX_WORLD_SIZE

echo -e $"\e[1mEND: Ring\e[0m\n"
######################### END: Ring ########################

# for (( K=1; K<=$MAX_WORLD_SIZE; K++ ))
# do
#     view-results topology --k $K --dataset $DATASET --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save 
# done
# view-results topology --dataset $DATASET --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save

# Clear cache
clean_dataset $DATASET
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"

echo -e $"\e[1mEND: Experiments\e[0m"
##################### END: Experiments #####################