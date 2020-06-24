#!/bin/bash
DATASET=${DATASET:-mg_scale}
K=${K:-6}
OVERSUB=${OVERSUB:-false}

colatools load $DATASET \
    split --k $K --seed 1 $DATASET

echo -e $"|-> Setting Parameters"
OUTPUT_DIR=${OUTPUT_DIR:-'./log'}
export JOBLIB_CACHE_DIR='./cache'

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


log_path=$OUTPUT_DIR'/'$DATASET'/'$K'/'
# Run cola
echo -e $"|-> Running CoLA, world size="$K
if [ $OVERSUB = true ] 
then
    mpirun -n $K --output-filename $log_path'mpilog' --oversubscribe run-cola \
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
        --dataset $DATASET \
        --solvername $local_alg \
        --algoritmname $global_alg \
        --use_split_dataset
else
    mpirun -n $K --output-filename $log_path'mpilog' run-cola \
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
        --dataset $DATASET \
        --solvername $local_alg \
        --algoritmname $global_alg \
        --use_split_dataset
fi
# Save result plot
echo -e $"|-> Saving result plots to 'out/"$DATASET"/"$K"/'"
viewresults --dataset $DATASET --k $K --no-show --save &> /dev/null;