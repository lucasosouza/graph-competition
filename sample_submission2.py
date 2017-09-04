# sample_submission.py

import pandas as pd
import numpy as np

#### import file

# import csv
f = "networks/real2.csv"
df = pd.DataFrame.from_csv(f)

#### get list of unique nodes

# get union of two sets, origin and destination
origin = set(df.index.unique()) 
destination = set(df['1'].unique()) 
nodes = origin | destination
len_nodes = len(nodes)

# convert to list
nodes = np.array((list(nodes)))

# shuffle
np.random.shuffle(nodes)

# insert identifier in the beggining
nodes = np.append('real2', nodes)

#### prepare, create and export dataframe

# create columns
cols = ['NetID']
for i in range(1, len_nodes+1):
    cols.append('nodeID' + str(i))

# reshape nodes to fit dataframe format
nodes = nodes.reshape((1, len_nodes+1))

# create dataframe
df = pd.DataFrame(nodes, columns=cols)

# export to csv
df.to_csv('test_submission2.csv', index=False)


