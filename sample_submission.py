# sample_submission.py

import pandas as pd
import random
import numpy as np

# create columns
cols = ['NetID']
for i in range(1,501):
    cols.append('nodeID' + str(i))

df = pd.DataFrame(np.zeros((8,501)), columns=cols, dtype=np.int32)

# create nodes
ct = 0
for graph_type in ['model', 'real']:
    for i in range(1,5):
        df.ix[ct, 'NetID'] = (graph_type + str(i))
        ct += 1

random_numbers = set()
for i in range(1000000):
    random_numbers.add(int(random.random()*1e6))

random_numbers = list(random_numbers)
np.random.shuffle(random_numbers)

# for every columns
for i in range(1,501):
    ref_col = ('nodeID' + str(i))
    # for every row
    for j in range(8):
        df.ix[j, ref_col] = random_numbers.pop()

df.to_csv('test_submission.csv', index=False)

