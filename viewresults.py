import csv
import numpy as np
import matplotlib.pyplot as plt

w = np.load('log/weight.npy', allow_pickle=True)
print('weights:')
k=0
for i in range(len(w)):
    print(f'\tNode {k}: {w[k]}')
    k+=1

print('norm sq of concensus violation:')
with open('log/0result.csv', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        print(f"\tIter {row['i_iter']}: {row['cv2']}")


r"""
Plots
1. Absolute CV2
2. Relative CV2

"""

