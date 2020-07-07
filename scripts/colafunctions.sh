export JOBLIB_CACHE_DIR='./cache'

pause() {
    read -n 1 -p "Press any key to continue..." > /dev/null  
}

pad() {
    echo $1 | awk '{printf "%02d\n", $0;}'
}

set_cola_parameters() {
    declare -g LOG_DIR=${LOG_DIR:-'./log'}
    declare -g OUT_DIR=${OUT_DIR:-'./out'}

    DATASET=${DATASET:-'rderms2'}
    declare -g MAX_WORLD_SIZE
    declare -g TRAIN_SIZE
    declare -g L1_RATIO
    declare -g LAMBDA
    declare -g LINREG_ITER
    if [[ $DATASET == 'rderms2'* ]]; then
        MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-16}
        TRAIN_SIZE=${TRAIN_SIZE:-24}
        L1_RATIO=${L1_RATIO:-.7}
        LAMBDA=${LAMBDA:-1e-3}
        # LINREG_ITER=${LINREG_ITER:-'--linreg-iter 1'}
    elif [[ $DATASET == 'rderms16'* ]]; then
        MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-16}
        TRAIN_SIZE=${TRAIN_SIZE:-16}
        L1_RATIO=${L1_RATIO:-0}
        # LINREG_ITER=${LINREG_ITER:-'--linreg-iter 1'}
        LAMBDA=${LAMBDA:-1e-4}
    elif [[ $DATASET == 'rderms' ]]; then
        MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-5}
        TRAIN_SIZE=${TRAIN_SIZE:-24}
        L1_RATIO=${L1_RATIO:-.2}
        LAMBDA=${LAMBDA:-1e-3}
        # LINREG_ITER=${LINREG_ITER:-'--linreg-iter 1'}
    elif [[ $DATASET == 'mg'* ]]; then 
        MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-6}
        TRAIN_SIZE=${TRAIN_SIZE:-0.7}
        LAMBDA=${LAMBDA:-5e-4}
        L1_RATIO=${L1_RATIO:-0.5}
    elif [[ $DATASET == 'housing'* ]]; then 
        MAX_WORLD_SIZE=${MAX_WORLD_SIZE:-13}
        TRAIN_SIZE=${TRAIN_SIZE:-0.6}
        LAMBDA=${LAMBDA:-1e-7}
        L1_RATIO=${L1_RATIO:-0.5}
    fi
    echo -e $"|-> Parameters"

    declare -g RANDOM_STATE=${RANDOM_STATE:-42}
    echo -e $"|---> random_state="$RANDOM_STATE

    declare -g GLOBAL_STEPS=${GLOBAL_STEPS:-200}
    echo -e $"|---> global_steps="$GLOBAL_STEPS
    echo -e $"|---> l1_ratio="$L1_RATIO
    echo -e $"|---> lambda="$LAMBDA
    declare -g THETA=${THETA:-1e-7}
    echo -e $"|---> theta="$THETA
    declare -g TOPOLOGY=${TOPOLOGY:-complete}
    echo -e $"|---> graph_topology="$TOPOLOGY

    declare -g LOCAL_ALG=${LOCAL_ALG:-ElasticNet}
    echo -e $"|-> Using local algorithm "$LOCAL_ALG
    declare -g GLOBAL_ALG=${GLOBAL_ALG:-cola}
    echo -e $"|-> Using global algorithm "$GLOBAL_ALG

    declare -g OVERSUB=${OVERSUB:-false}
    if [ $OVERSUB = true ]; then
        OVERSUB_FLAG='--oversubscribe';
    else
        OVERSUB_FLAG=''
    fi
    declare -g VERBOSE=${V:-1}
    if [[ $VERBOSE -gt 1 ]]; then
        VERBOSE_FLAG='-v';
    else
        VERBOSE_FLAG=''
        export PYTHONWARNINGS="ignore"
    fi
}

clean_dataset() {
    local DATASET=$1
    local ALL=${2:-false}

    echo -e $"|-> Cleaning split files for '"$DATASET"'"
    rm -rf ./data/$DATASET &> /dev/null
    if [ $ALL = true ]; then
        echo -e $"|-> Cleaning log and output for '"$DATASET"'"
        rm -rf $LOG_DIR/$DATASET &> /dev/null
        rm -rf $OUT_DIR/$DATASET &> /dev/null
    fi
}

run_cola_n() {
    local DATASET=$1
    local N=${3:-false}
    declare START
    if [ $N = false ]; then
        N=$2; START=1
    else
        START=$2
    fi
    echo "$DATASET, $START, $N"

    for (( WORLD_SIZE=$START; WORLD_SIZE<=$N; WORLD_SIZE++ ))
    do
        run_cola $DATASET $WORLD_SIZE
    done
}

run_cola() {
    local DATASET=$1
    K_l=$2
    LOG_PATH_l="$LOG_DIR/$DATASET/"$(pad $K_l)"/$TOPOLOGY"
    # Run cola
    echo -e $"|-> Running CoLA, world size=$WORLD_SIZE, oversubscribe=${OVERSUB_FLAG}"
    echo -e $"|---> Logging to '$LOG_PATH_l'"
    mpirun -n $K_l --output-filename $LOG_PATH_l'/mpilog' ${OVERSUB_FLAG} run-cola \
        --split_by 'features' \
        --max_global_steps $GLOBAL_STEPS \
        --graph_topology $TOPOLOGY \
        --exit_time 1000.0 \
        --logmode='all' \
        --theta $THETA \
        --l1_ratio $L1_RATIO \
        --lambda_ $LAMBDA \
        --output_dir $LOG_DIR \
        --dataset_size 'all' \
        --ckpt_freq 1 \
        --local_iters 25 \
        --dataset $DATASET \
        --solvername $LOCAL_ALG \
        --algoritmname $GLOBAL_ALG \
        --use_split_dataset \
        --random_state $RANDOM_STATE \
        --verbose $VERBOSE 
    # Save result plot
    echo -e $"|---> Saving result plots to '$OUT_DIR/$DATASET/$K_l/$TOPOLOGY/', linreg_iter=$LINREG_ITER"
    view-results plot-results --dataset $DATASET $LINREG_ITER --topology $TOPOLOGY --k $K_l --logdir $LOG_DIR --savedir $OUT_DIR --no-show --save --large;
}