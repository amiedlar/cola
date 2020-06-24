#!/bin/sh

replace_dataset="mg_scale_replace"
insert_dataset="mg_scale_insert"
control_dataset="mg_scale"

export JOBLIB_CACHE_DIR='./cache'

clean_dataset() {
    DATASET=$1
    ALL=${2:-false}

    echo -e $"|-> Cleaning split files for '"$DATASET"'"
    rm -rf ./data/$DATASET &> /dev/null
    if [ $ALL = true ]; then
        echo -e $"|-> Cleaning log and output for '"$DATASET"'"
        rm -rf ./log/$DATASET &> /dev/null
        rm -rf ./out/$DATASET &> /dev/null
    fi
}

echo -e $"Setting up datasets"
rm -rf cache
clean_dataset $replace_dataset true;
echo -e $"\e[2m"
colatools load mg_scale \
    replace-column --scale-col 0 --scale-by 1.1 5 scale \
    replace-column --scale-col 1 --scale-by .9  4 scale \
    replace-column --weights "0.5 0.5 0 0 0" 3 weights \
    replace-column 2 uniform \
    split $replace_dataset
echo -e $"\e[0m"
clean_dataset $insert_dataset true;
echo -e $"\e[2m"
colatools load mg_scale \
    insert-column --scale-col 0 --scale-by 1.1 scale \
    insert-column --scale-col 1 --scale-by 0.9 scale \
    insert-column --weights "1 2 3 4 5 6" weights \
    insert-column uniform \
    split $insert_dataset
echo -e $"\e[0m"
clean_dataset $control_dataset true;
echo -e $"\e[2m"
colatools load mg_scale \
    split --seed 1 $control_dataset
echo -e $"\e[0m"
echo -e $"Starting experiments..."

echo -e $"|-> Setting Parameters"
export OUTPUT_DIR='./log'

global_steps=200
echo -e $"|--> global_steps="$global_steps
l1_ratio=0.5
echo -e $"|--> l1_ratio="$l1_ratio
lambda=1e-2
echo -e $"|--> lambda="$lambda
theta=1e-7
echo -e $"|--> theta="$theta
topology='complete'
echo -e $"|--> graph_topology="$topology

local_alg='ElasticNet'
echo -e $"|-> Using local algorithm "$local_alg
global_alg='cola'
echo -e $"|-> Using global algorithm "$global_alg

echo
echo -e $"\e[1mSTART: Control\e[0m"

dataset=$control_dataset
for world_size in {1..6};
do
    log_path=$OUTPUT_DIR'/'$dataset'/'$world_size'/'
    # Run cola
    echo -e $"|-> Running CoLA, world size="$world_size
    mpirun -n $world_size --output-filename $log_path'mpilog' --oversubscribe run-cola \
        --split_by 'features' \
        --max_global_steps $global_steps \
        --graph_topology $topology \
        --exit_time 1000.0 \
        --logmode='all' \
        --theta $theta \
        --l1_ratio $l1_ratio \
        --lambda_ $lambda \
        --output_dir ${OUTPUT_DIR} \
        --dataset_size 'all' \
        --ckpt_freq 1 \
        --dataset $dataset \
        --solvername $local_alg \
        --algoritmname $global_alg \
        --use_split_dataset \
    	&> /dev/null;
    # Save result plot
    echo -e $"|-> Saving result plots to 'out/"$dataset"/"$world_size"/'.."
    viewresults --dataset $dataset --k $world_size --no-show --save &> /dev/null;
done;

# Clean up
clean_dataset $dataset false
echo -e $"\e[1mEND: Control\e[0m\n"

echo -e $"\e[1mSTART: Column Replacement\e[0m"

dataset=$replace_dataset

for world_size in {1..6};
do
    log_path=$OUTPUT_DIR'/'$dataset'/'$world_size'/'
    # Run cola
    echo -e $"|-> Running CoLA, world size="$world_size
    mpirun -n $world_size --output-filename $log_path'mpilog' --oversubscribe run-cola \
        --split_by 'features' \
        --max_global_steps $global_steps \
        --graph_topology $topology \
        --exit_time 1000.0 \
        --logmode='all' \
        --theta $theta \
        --l1_ratio $l1_ratio \
        --lambda_ $lambda \
        --output_dir ${OUTPUT_DIR} \
        --dataset_size 'all' \
        --ckpt_freq 1 \
        --dataset $dataset \
        --solvername $local_alg \
        --algoritmname $global_alg \
        --use_split_dataset \
    	&> /dev/null;
    # Save result plot
    echo -e $"|-> Saving result plots to 'out/"$dataset"/"$world_size"/'.."
    viewresults --dataset $dataset --k $world_size --no-show --save &> /dev/null;
done;

# Clean up
clean_dataset $dataset false
echo -e $"\e[1mEND: Column Replacement\e[0m\n"

echo -e $"\e[1mSTART: Column Insertion\e[0m"

dataset=$insert_dataset
for world_size in {1..10};
do
    log_path=$OUTPUT_DIR'/'$dataset'/'$world_size'/'
    # Run cola
    echo -e $"|-> Running CoLA, world size="$world_size
    mpirun -n $world_size --output-filename $log_path'mpilog' --oversubscribe run-cola \
        --split_by 'features' \
        --max_global_steps $global_steps \
        --graph_topology $topology \
        --exit_time 1000.0 \
        --logmode='all' \
        --theta $theta \
        --l1_ratio $l1_ratio \
        --lambda_ $lambda \
        --output_dir ${OUTPUT_DIR} \
        --dataset_size 'all' \
        --ckpt_freq 1 \
        --dataset $dataset \
        --solvername $local_alg \
        --algoritmname $global_alg \
        --use_split_dataset \
    	&> /dev/null;
    # Save result plot
    echo -e $"|-> Saving result plots to 'out/"$dataset"/"$world_size"/'.."
    viewresults --dataset $dataset --k $world_size --no-show --save &> /dev/null;
done;

# Clean up
clean_dataset $dataset false
echo -e $"\e[1mEND: Column Insertion\e[0m\n"
rm -rf cache