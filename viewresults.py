import numpy as np

w = np.load('log/weight.npy', allow_pickle=True)
print('weights:')
k=0
for i in range(len(w)):
    print(f'\tNode {k}: {w[k]}')
    k+=1

import csv
print('norm sq of concensus violation:')
with open('log/0result.csv', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        print(f"\tIter {row['i_iter']}: {row['cv2']}")
