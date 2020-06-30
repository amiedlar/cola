#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters

##################### START: Load Data #####################
echo -e $"\e[1mSTART: Load Data\e[0m"

CONTROL_DATASET=$DATASET
REPLACE_DATASET=$DATASET"_replace1"
INSERT_DATASET=$DATASET"_insert1"
MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-16}

rm -rf $JOBLIB_CACHE_DIR

clean_dataset $DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    split --train 4 $CONTROL_DATASET 
echo -e $"\e[0m"

clean_dataset $REPLACE_DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    replace-column 5 uniform \
    split $REPLACE_DATASET
echo -e $"\e[0m"

clean_dataset $INSERT_DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG}\
    load $DATASET \
    insert-column uniform \
    split $INSERT_DATASET
echo -e $"\e[0m"

echo -e $"\e[0m\e[1mEND: Load Data\e[0m\n"
###################### END: Load Data ######################

#################### START: Experiments ####################
echo -e $"\e[1mSTART: Rank Experiments\e[0m"

###################### START: Control ######################
echo -e $"\e[1mSTART: Control\e[0m"

# Run CoLA 
run_cola_n $CONTROL_DATASET $MAX_WORLD_SIZE

# Clean up
clean_dataset $CONTROL_DATASET

echo -e $"\e[1mEND: Control\e[0m\n"
####################### END: Control #######################

#################### START: Replacement ####################
echo -e $"\e[1mSTART: Column Replacement\e[0m"

# Run CoLA
run_cola_n $REPLACE_DATASET $MAX_WORLD_SIZE

# Clean up
clean_dataset $REPLACE_DATASET

echo -e $"\e[1mEND: Column Replacement\e[0m\n"
##################### END: Replacement #####################

##################### START: Insertion #####################
echo -e $"\e[1mSTART: Column Insertion\e[0m"

# Run CoLA
run_cola_n $INSERT_DATASET $MAX_WORLD_SIZE

# Clean up
clean_dataset $INSERT_DATASET

echo -e $"\e[1mEND: Column Insertion\e[0m\n"
###################### END: Insertion ######################

# Clear cache
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"
echo -e $"\e[1mEND: Rank Experiments\e[0m"
##################### END: Experiments #####################
