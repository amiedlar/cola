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

######################## START: Grid #######################
echo -e $"\e[1mSTART: Grid\e[0m"

TOPOLOGY='grid' 
# Run CoLA 
run_cola $DATASET 4
run_cola $DATASET 16

echo -e $"\e[1mEND: Grid\e[0m\n"
######################### END: Grid ########################

for (( K=1; K<=$MAX_WORLD_SIZE; K++ ))
do
    view-results topology --k $K --dataset $DATASET --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save
done

# Clear cache
clean_dataset $DATASET
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"

echo -e $"\e[1mEND: Experiments\e[0m"
##################### END: Experiments #####################