replace_dataset="mg_scale_replace"
insert_dataset="mg_scale_replace"


colatools load mg_scale \
    replace-column --scale-col 0 --scale-by 1.1 5 scale \
    replace-column --scale-col 1 --scale-by .9  4 scale \
    replace-column --weights "-0.1 -0.2 2 0 0" 3 weights \
    replace-column 2 uniform \
    split $replace_dataset

colatools load mg_scale \
    insert-column --scale-col 0 --scale-by 1.1 scale \
    insert-column --scale-col 1 --scale-by 0.9 scale \
    insert-column --weights "1 2 3 4 5 6" weights \
    insert-column uniform \
    split $insert_dataset


echo "Starting experiments..."

echo "--> Setting Parameters"
export JOBLIB_CACHE_DIR='./cache'
export OUTPUT_DIR='./log'

global_steps=20
l1_ratio=1.0
lambda=1e-4
theta=1e-7

local_alg='ElasticNet'
global_alg='cola'

topology='complete'

echo "Rank: Column Replacement"
dataset=$replace_dataset
for world_size in {1..6};
do
    log_path=${OUTPUT_DIR}'/'$dataset'/'$world_size'/'
    # Run cola
    echo "|-> Running CoLA, world size="$world_size
    mpirun -n $world_size -output-filename $logpath'mpilog' --oversubscribe python3 run_cola.py \
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
        --use_split_dataset
    # Save result plot
    echo "|-> Saving result plots to 'out/"$dataset"/"$world_size"/'.."
    viewresults --dataset $dataset --k $world_size --no-show --save