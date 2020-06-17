# Experimental Results of CoLa and CoCoA
## RDerms Data
### 1 Node
```
mpirun -n 1 --oversubscribe python3 run_cola.py \ 
    --split_by 'features' \ 
    --max_global_steps 200 \ 
    --graph_topology 'complete' \ 
    --exit_time 1000.0 \ 
    --logmode='all' \ 
    --theta 1e-7 \ 
    --l1_ratio 0 \ 
    --lambda_ 0 \ 
    --output_dir ${OUTPUT_DIR} \ 
    --dataset_size 'all' \ 
    --ckpt_freq 2 \ 
    --solvername ElasticNet \ 
    --algoritmname cola \ 
    --use_split_dataset \ 
    --dataset 'rderms'
```