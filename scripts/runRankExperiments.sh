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
VERBOSE=${V:-false}
if [ $VERBOSE = true ]; then
    VERBOSE_FLAG='-v';
else
    VERBOSE_FLAG=''
fi

run_cola() {
    dataset=$1
    n=$2

    for (( world_size=1; world_size<=$n; world_size++ ))
    do
        log_path=$LOG_DIR'/'$dataset'/'$world_size'/'
        # Run cola
        echo -e $"|-> Running CoLA, world size=$world_size, oversubscribe=${OVERSUB_FLAG}"
        echo -e $"|---> Logging to '$log_path'"
        mpirun -n $world_size --output-filename $log_path'mpilog' --report-bindings ${OVERSUB_FLAG} run-cola \
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
            --dataset $dataset \
            --solvername $local_alg \
            --algoritmname $global_alg \
            --use_split_dataset \
            > /dev/null;
        # Save result plot
        echo -e $"|---> Saving result plots to '"$OUT_DIR"/"$dataset"/"$world_size"/'"
        viewresults --dataset $dataset --k $world_size --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save &> /dev/null;
    done
}


##################### START: Load Data #####################
echo -e $"\e[1mSTART: Load Data\e[0m"

replace_dataset="mg_scale_replace1"
insert_dataset="mg_scale_insert1"
control_dataset="mg_scale"
permute_dataset="mg_scale_permute"

rm -rf cache
clean_dataset $replace_dataset true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load mg_scale \
    replace-column 5 uniform \
    split $replace_dataset
    # replace-column --scale-col 0 --scale-by 1.1 5 scale \
    #replace-column --scale-col 1 --scale-by .9  4 scale \
    #replace-column --weights "0.5 0.5 0 0 0" 3 weights \
echo -e $"\e[0m"
clean_dataset $insert_dataset true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG}\
    load mg_scale \
    insert-column uniform \
    split $insert_dataset
    # insert-column --scale-col 0 --scale-by 1.1 scale \
    #insert-column --scale-col 1 --scale-by 0.9 scale \
    #insert-column --weights "1 2 3 4 5 6" weights \
echo -e $"\e[0m"
clean_dataset $control_dataset true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load mg_scale \
    split $control_dataset
echo -e $"\e[0m"
clean_dataset $control_dataset true;
echo -e $"\e[2m"
colatools ${VERBOSE_FLAG} \
    load mg_scale \
    split --seed 42 $permute_dataset
echo -e $"\e[0m\e[1mEND: Load Data\e[0m\n"
###################### END: Load Data ######################

#################### START: Experiments ####################
echo -e $"\e[1mSTART: Rank Experiments\e[0m"

echo -e $"|-> Setting Parameters"

LOG_DIR=${LOG_DIR:-'./log'}
OUT_DIR=${OUT_DIR:-'./out'}
export JOBLIB_CACHE_DIR='./cache'

global_steps=${global_steps:-200}
echo -e $"|---> global_steps="$global_steps
l1_ratio=${l1_ratio:-0.5}
echo -e $"|---> l1_ratio="$l1_ratio
lambda=${lambda:-1e-2}
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
echo -e $"\e[1mSTART: Control\e[0m"

# Run CoLA
run_cola $control_dataset 6

# Clean up
clean_dataset $control_dataset

echo -e $"\e[1mEND: Control\e[0m\n"
####################### END: Control #######################

#################### START: Replacement ####################
echo -e $"\e[1mSTART: Column Replacement\e[0m"

# Run CoLA
run_cola $replace_dataset 6

# Clean up
clean_dataset $replace_dataset

echo -e $"\e[1mEND: Column Replacement\e[0m\n"
##################### END: Replacement #####################

##################### START: Insertion #####################
echo -e $"\e[1mSTART: Column Insertion\e[0m"

# Run CoLA
run_cola $insert_dataset 7

# Clean up
clean_dataset $insert_dataset

echo -e $"\e[1mEND: Column Insertion\e[0m\n"
###################### END: Insertion ######################

# Clear cache
rm -rf $JOBLIB_CACHE_DIR
echo -e "\e[1mCleared $JOBLIB_CACHE_DIR\e[0m"
##################### END: Experiments #####################
