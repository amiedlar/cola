#!/bin/bash
clean_dataset() {
    DATASET=$1
    ALL=${2:-false}

    echo -e $"|-> Cleaning split files for '"$DATASET"'"
    rm -rf ./data/$DATASET &> /dev/null
    if [ $ALL = true ]; then
        echo -e $"|-> Cleaning log and output for '"$DATASET"'"
        rm -rf $LOG_DIR/$DATASET &> /dev/null
        rm -rf $OUT_DIR/$DATASET &> /dev/null
    fi
}

OVERSUB=${OVERSUB:-false}
if [ $OVERSUB = true ]; then
    OVERSUB_FLAG='--oversubscribe';
else
    OVERSUB_FLAG=''
fi
VERBOSE=${V:-1}
if [ $VERBOSE > 1 ]; then
    VERBOSE_FLAG='-v';
else
    VERBOSE_FLAG=''
fi

run_cola() {
    dataset=$1
    n=$2
    start=${3:-1}

    for (( world_size=$start; world_size<=$n; world_size++ ))
    do
        log_path=$LOG_DIR'/'$dataset'/'$world_size'/'
        # Run cola
        echo -e $"|-> Running CoLA, world size=$world_size, oversubscribe=${OVERSUB_FLAG}"
        echo -e $"|---> Logging to '$log_path'"
        mpirun -n $world_size --output-filename $log_path'mpilog' ${OVERSUB_FLAG} run-cola \
            --split_by 'features' \
            --max_global_steps $global_steps \
            --graph_topology $topology \
            --exit_time 1000.0 \
            --logmode='all' \
            --theta $theta \
            --l1_ratio $l1_ratio \
            --lambda_ $lambda \
            --output_dir ${LOG_DIR} \
            --dataset_size 'all' \
            --ckpt_freq 1 \
            --local_iters 10 \
            --dataset $dataset \
            --solvername $local_alg \
            --algoritmname $global_alg \
            --use_split_dataset \
            --verbose $VERBOSE 
        # Save result plot
        echo -e $"|---> Saving result plots to '"$OUT_DIR"/"$dataset"/"$world_size"/'"
        viewresults --dataset $dataset --k $world_size --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save &> /dev/null;
    done
}

DATASET=${DATASET:-'rderms2'}
MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-16}

clean_dataset $DATASET true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load $DATASET \
    split --train 4 $DATASET 
echo -e $"\e[0m"

#################### START: Experiments ####################
echo -e $"\e[1mSTART: Experiments\e[0m"

echo -e $"|-> Setting Parameters"

LOG_DIR=${LOG_DIR:-'./log'}
OUT_DIR=${OUT_DIR:-'./out'}
export JOBLIB_CACHE_DIR='./cache'

global_steps=${global_steps:-1000}
echo -e $"|---> global_steps="$global_steps
l1_ratio=${l1_ratio:-0.5}
echo -e $"|---> l1_ratio="$l1_ratio
lambda=${lambda:-1e-4}
echo -e $"|---> lambda="$lambda
theta=${theta:-1e-7}
echo -e $"|---> theta="$theta
topology=${topology:-complete}
echo -e $"|---> graph_topology="$topology


local_alg=${local_alg:-ElasticNet}
echo -e $"|-> Using local algorithm "$local_alg
global_alg=${global_alg:-cola}
echo -e $"|-> Using global algorithm "$global_alg

echo
###################### START: Control ######################
echo -e $"\e[1mSTART: CoLA\e[0m"

# Run CoLA 
run_cola $DATASET $MAX_WORLD_SIZE

# Clean up
clean_dataset $DATASET

echo -e $"\e[1mEND: CoLA\e[0m\n"
####################### END: Control #######################

# Clear cache
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"
echo -e $"\e[1mEND: Experiments\e[0m"
##################### END: Experiments #####################