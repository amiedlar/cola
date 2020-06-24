## Partitioning and Running

From the cola-master directory: Use CoLATools to load the dataset and split across ranks
```
svm_source_dir=../data
local_data_dir=./data
dataset=rderms2
colatools --indir $svm_source_dir --outdir $local_data_dir \
    load $dataset \
    split $dataset
```

To run cola:
```
export JOBLIB_CACHE_DIR='./cache'
export OUTPUT_DIR='./log'
DATASET_PATH='./data/rderms/features/'$world_size

mpirun -n $world_size --oversubscribe python3 run_cola.py \
    --split_by 'features' \
    --max_global_steps 20 \
    --graph_topology 'complete' \
    --exit_time 1000.0 \
    --logmode='all' \
    --theta 1e-7 \
    --l1_ratio 0 \
    --lambda_ 0 \
    --output_dir ${OUTPUT_DIR} \
    --dataset_size 'all' \
    --ckpt_freq 2 \
    --dataset mg_scale \
    --solvername LinearRegression \
    --algoritmname cola \
    --use_split_dataset
```


## CoLATools Rank Manipulation Example
```
colatools load mg_scale \ 
    replace-column --scale-col 0 --scale-by 1.1 5 scale \
    replace-column --scale-col 1 --scale-by .9  4 scale \
    replace-column --weights "[-0.1, -0.2, 2, 0, 0]" 3 weight \ 
    replace-column 2 uniform \ 
    save-svm mg_scale_replace.svm
```
