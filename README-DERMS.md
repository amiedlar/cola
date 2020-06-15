## Partitioning and Running

From the cola-master directory:
```
world_size=4
python3 split_dataset.py --K=$world_size --input_file=~/working/data/data.svm --outdir=data/rderms
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
    --l1_ratio 0.5 \
    --lambda_ 1e-4 \
    --output_dir ${OUTPUT_DIR} \
    --dataset_size 'all' \
    --ckpt_freq 2 \
    --dataset_path ${DATASET_PATH} \
    --solvername ElasticNet \
    --algoritmname cola \
    --use_split_dataset
```
from sklearn.datasets import dump_svmlight_file
import numpy as np
X = np.loadtxt('../../data/data.txt')
dump_svmlight_file(X[:,0:5], X[:,0],'data/data1.svm')