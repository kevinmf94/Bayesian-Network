import numpy as np
import pandas
import math
import time
import BayesianNetwork as bn

#FOR RANDOM PERMUTATIONS ---> df.iloc[np.random.permutation(len(df))]

"""
Splits the data in two dataFrames, one of trainPercent size of the original and other of 1-trainPercent size

"""
def splitData(dataBase, trainPercent=0.66):
	assert 0<trainPercent<1

	dataBaseLength = dataBase.shape[0]
	trainLength = int(dataBaseLength * float(trainPercent))
	train = dataBase.loc[:trainLength,:]
	validation = dataBase.loc[trainLength:,:]
	return train, validation

"""
This function split the dataframe mantaining exactly the same proportion of labels than the original.
This function mantaining too the shuffle or non-shuffle critery of the dataFrame
"""
def splitDataInProportion(dataBase, trainPercent = 0.66):
	assert 0 < trainPercent < 1

	negativeIndexs = dataBase[dataBase['sentimentLabel'] == 0].index.values
	positiveIndexs = dataBase[dataBase['sentimentLabel'] == 1].index.values
	negativesToCatch = int(len(negativeIndexs)*trainPercent)
	positivesToCatch = int(len(positiveIndexs)*trainPercent)
	trainIndex = np.append(negativeIndexs[:negativesToCatch], positiveIndexs[:positivesToCatch])
	validationIndex = np.append(negativeIndexs[negativesToCatch:], positiveIndexs[positivesToCatch:])
	train = dataBase.loc[trainIndex,:]
	validation = dataBase.loc[validationIndex,:]
	return train, validation

"""
Gets a list of size nFolds with the indexes of diferent folds for apply a kfold validation
"""
def kFold(dataBase, nFolds=5):
	assert 1<nFolds <= dataBase.shape[0]

	dataBaseLength = dataBase.shape[0]
	nRowsGroup = int(dataBaseLength / nFolds)
	folds = [range((n * nRowsGroup),((n+1) * nRowsGroup)) for n in range(nFolds)]
	return folds

