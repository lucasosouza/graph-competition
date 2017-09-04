import os
import sys
import math

import numpy
import pandas
import pickle

# Generalized matrix operations:

def __extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes

def __makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()
    
    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix) # insert missing columns
    matrix = insertMissingColumns(matrix.T).T # insert missing rows

    return matrix.fillna(default)

def __ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(numpy.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T

def __normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)

def __euclideanNorm(series):
    return math.sqrt(series.dot(series))

# PageRank specific functionality:

def __startState(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    startProb = 1.0 / float(len(nodes))
    return pandas.Series({node : startProb for node in nodes})

def __integrateRandomSurfer(nodes, transitionProbs, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transitionProbs.copy().multiply(1.0 - rsp) + alpha

def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    # Clerical work:
    # transitionWeights = pandas.DataFrame(transitionWeights)
    nodes = __extractNodes(transitionWeights)
    print("transition started")
    transitionWeights = __makeSquare(transitionWeights, nodes, default=0.0)
    transitionWeights = __ensureRowsPositive(transitionWeights)
    print("transition finished")

    # Setup:
    state = __startState(nodes)
    transitionProbs = __normalizeRows(transitionWeights)
    transitionProbs = __integrateRandomSurfer(nodes, transitionProbs, rsp)
    print("setup finished")

    
    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbs)
        delta = state - oldState
        # print(delta)
        if __euclideanNorm(delta) < epsilon: break

    return state

def to_matrix(origin_df):

    #ceil = 1000

    # get unique nodes
    source = set(origin_df.index.unique()) 
    target = set(origin_df['1'].unique()) 
    nodes = source | target  
    # nodes = list(nodes)[:ceil] 

    # create zero matrix and init dataframe
    len_nodes = len(nodes)
    # len_nodes = ceil 
    zero_matrix = numpy.zeros((len_nodes, len_nodes))  
    new_df = pandas.DataFrame(zero_matrix, index=nodes, columns=nodes, dtype=numpy.int32)

    # itera e cria a matriz
    for i in range(len_nodes):
        # filtra casos que nao estao no index
        if i in source:
            # cria a lista com os target/outgoing
            outgoing_nodes = numpy.array(origin_df.ix[i, '1']).flatten()
            # itero pelo target
            for j in [var for var in outgoing_nodes]: # if var < ceil]:
                # atualizo o valor na matriz
                new_df.ix[i, j] = 1

    # print(numpy.bincount(new_df.ix[:, 370] > 0))
    return new_df

f = "networks/real2.csv"
# out_f = "input_test.csv"
out_df = to_matrix(pandas.DataFrame.from_csv(f))
# out_df.to_csv(out_f)

# f = "input_test.csv"
# in_df = pandas.DataFrame.from_csv(f, dtypes=numpy.int32)
# in_df = pandas.read_csv(f, dtype=numpy.int32, index_col=0)
# import pdb;pdb.set_trace()
state = powerIteration(out_df)

print(state.order(ascending=False))
with open('page_rank_results.p', 'wb') as handle:
    pickle.dump(state, handle)
