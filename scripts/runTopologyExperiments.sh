#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters
TRAIN_SIZE=${TRAIN_SIZE:-0.7}
#################### START: Experiments ####################
echo -e $"\e[1mSTART: Experiments\e[0m"
echo -e $"\e[2m"
# if [[ $SCALE = true ]]; then
# clean_dataset $DATASET'_scale' true;
# colatools ${VERBOSE_FLAG} \
#     load $DATASET \
#     scale \
#     info-rank \
#     info-cond --p 'fro' \
#     info-cond --p 1 \
#     info-cond --p 2 \
#     dump-svm --overwrite $DATASET'_scale' \
#     split --train $TRAIN_SIZE --seed $RANDOM_STATE $DATASET'_scale'
# DATASET=$DATASET'_scale'
# else
clean_dataset $DATASET true;
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    info-rank \
    info-cond \
    split --train $TRAIN_SIZE --seed $RANDOM_STATE $DATASET
# fi
# echo -e $"\e[0m"
# ###################### START: Complete #####################
# echo -e $"\e[1mSTART: Complete\e[0m"

TOPOLOGY='complete' 
# Run CoLA 
run_cola_n $DATASET 8 $MAX_WORLD_SIZE

# echo -e $"\e[1mEND: Complete\e[0m\n"
# ####################### END: Complete ######################

# ######################## START: Ring #######################
# echo -e $"\e[1mSTART: Ring\e[0m"

TOPOLOGY='ring' 
# Run CoLA 
run_cola_n $DATASET 8 $MAX_WORLD_SIZE

# echo -e $"\e[1mEND: Ring\e[0m\n"
# ######################### END: Ring ########################

# ######################## START: Grid #######################
# echo -e $"\e[1mSTART: Grid\e[0m"

# TOPOLOGY='grid' 
# # Run CoLA 
# # run_cola $DATASET 4
# if [[ $MAX_WORLD_SIZE -ge 16 ]]; then
#     run_cola $DATASET 16
# fi

# echo -e $"\e[1mEND: Grid\e[0m\n"
######################## END: Grid ########################

for (( K=8; K<=$MAX_WORLD_SIZE; K+=4 ))
do
    view-results topology --k $K --dataset $DATASET --large --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save 
done
view-results topology --dataset $DATASET --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save

# Clear cache
clean_dataset $DATASET
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"

echo -e $"\e[1mEND: Experiments\e[0m"
##################### END: Experiments #####################