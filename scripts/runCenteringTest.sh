#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters

###################### START: Control ######################
echo -e $"\e[1mSTART: Fit Intercept\e[0m"

clean_dataset $DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    split --train $TRAIN_SIZE --seed $RANDOM_STATE $DATASET
echo -e $"\e[0m"
FIT_INTERCEPT='--fit-intercept'
GLOBAL_STEPS=1500
# run cola
run_cola $DATASET 16

# Clean up
clean_dataset $DATASET

echo -e $"\e[1mEND: Fit Intercept\e[0m\n"
####################### END: Control #######################

#################### START: Replacement ####################
echo -e $"\e[1mSTART: Ones \e[0m"

clean_dataset $DATASET-ones true
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    insert-column ones \
    dump-svm --overwrite $DATASET-ones \
    split --train $TRAIN_SIZE --seed $RANDOM_STATE $DATASET-ones
((MAX_WORLD_SIZE++))
echo -e $"\e[0m"
DATASET=$DATASET-ones

FIT_INTERCEPT='--no-fit-intercept'
# Run CoLA

run_cola $DATASET 17

# Clean up
clean_dataset $DATASET

echo -e $"\e[1mEND: Ones \e[0m\n"
##################### END: Replacement #####################