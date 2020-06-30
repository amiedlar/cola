#!/bin/bash
source scripts/colafunctions.sh
set_cola_parameters

#################### START: Experiments ####################
echo -e $"\e[1mSTART: Experiments\e[0m"
clean_dataset $DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    split --train 4 $DATASET 
echo -e $"\e[0m"

###################### START: Control ######################
echo -e $"\e[1mSTART: CoLA\e[0m"

# Run CoLA 
run_cola_n $DATASET $MAX_WORLD_SIZE

# Clean up
clean_dataset $DATASET

echo -e $"\e[1mEND: CoLA\e[0m\n"
####################### END: Control #######################

# Clear cache
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"

echo -e $"\e[1mEND: Experiments\e[0m"
##################### END: Experiments #####################