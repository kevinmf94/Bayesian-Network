import pylab as pl
import ErrorMeasurements as em
import BayesianNetwork as bn
import DataTrimmers as dt
import time
import numpy as np
import matplotlib.patches as mpatches

"""
Makes a basic cross validation with the specified percent of train
"""
def haveBasicCrossValidation(trainPercent = 0.66, parseDates=False, smooth = 0, data = None, maxWordsInDict = None, limitIncluded = True):
	if data is None:
		data = bn.readDatabase(shuffle = True)
	train, validation = dt.splitData(dataBase=data, trainPercent=trainPercent)
	bayesianNet = bn.BayesianNetwork(dataBase=train, parseDates=parseDates,maxWordsInDict=maxWordsInDict, limitIncludeInWordsDict=limitIncluded)
	timeStart = time.clock()
	confusionMatrix = bayesianNet.haveValidation(validationSet=validation, smooth=smooth)
	timeOfValidation = time.clock() - timeStart
	print "Time of Validation: " + str(timeOfValidation)
	return confusionMatrix

"""
Gets the error of one fold (useful for Kfold method)
"""
def getErrorOfOneFold(trainData,validationData, smooth = 0, maxWordsInDict = None, limitIncluded = True, confusionMatrix = [[0,0],[0,0]]):
	bayesianNet = bn.BayesianNetwork(dataBase=trainData, parseDates=False, maxWordsInDict=maxWordsInDict,
									 limitIncludeInWordsDict=limitIncluded, confusionMatrix=confusionMatrix)
	timeStart = time.clock()
	confusionMatrix = bayesianNet.haveValidation(validationSet=validationData, smooth=smooth)
	timeOfValidation = time.clock() - timeStart
	print "Time of Validation k: " + str(timeOfValidation)
	return confusionMatrix

"""
Have a K-fold validation using the defined number of K
"""
def haveKFoldValidation(k = 5,percentOfData=1, parseDates = False, smooth=0, data = None, maxWordsInDict = None, limitIncluded=True):
	if data is None:
		data = bn.readDatabase(shuffle=True)
	if percentOfData<1:
		data,_ = dt.splitData(data,trainPercent=percentOfData)
	confusionMatrix = [[0,0],[0,0]]
	folds = dt.kFold(data, nFolds=k)
	for i in range(len(folds)):
		trainData = data.drop(folds[i])
		validationData = data.loc[folds[i],:]
		confusionMatrix = getErrorOfOneFold(trainData = trainData, validationData=validationData, smooth=smooth,
									  maxWordsInDict=maxWordsInDict,limitIncluded=limitIncluded,
									  confusionMatrix=confusionMatrix)
	return confusionMatrix

"""
Have a Leave One Out validation with te specified percentOfData (100% by default)
"""
def leaveOneOutValidation(data = None, parseDates = False, smooth=0, percentOfData=1, maxWordsInDict = None, filter = []):
	assert 0<=smooth<=1
	assert 0<percentOfData<=1

	if data is None:
		data = bn.readDatabase(shuffle=True,filter=filter)
	if percentOfData < 1:
		data,_ = dt.splitData(dataBase=data, trainPercent=percentOfData)
	bayesianNet = bn.BayesianNetwork(dataBase=data, parseDates=False,maxWordsInDict=maxWordsInDict)
	timeStart = time.clock()
	confusionMatrix = bayesianNet.haveLeaveOneOutValidation(smooth=smooth)
	timeOfValidation = time.clock() - timeStart
	print "Leave one out validated in: " + str(timeOfValidation)
	return confusionMatrix

def CrossValidationMedian(trainPercent=0.66, parseDates=False, smooth=0,executions = 5):
	confusionMatrixs = []
	accuracy = []
	precision0= []
	precision1=[]
	recall0=[]
	recall1=[]
	for i in range(executions):
		confusionMatrix = haveBasicCrossValidation(trainPercent=trainPercent, parseDates=parseDates, smooth=smooth)
		accuracy.append(em.getAccuracy(confusionMatrix))
		precision0.append(em.getPrecision(confusionMatrix, positiveClass=0))
		precision1.append(em.getPrecision(confusionMatrix, positiveClass=1))
		recall0.append(em.getRecall(confusionMatrix, positiveClass=0))
		recall1.append(em.getRecall(confusionMatrix, positiveClass=1))
		confusionMatrixs.append(confusionMatrix)
	confusionMatrix = [[[],[]],[[],[]],]
	for i in range(executions):
		confusionMatrix[0][0].append(confusionMatrixs[i][0][0])
		confusionMatrix[0][1].append(confusionMatrixs[i][0][1])
		confusionMatrix[1][0].append(confusionMatrixs[i][1][0])
		confusionMatrix[1][1].append(confusionMatrixs[i][1][1])
	print "confusionMatrix of Means: "
	print "      Expect=0 Expect=1\n" \
		  "Pred=0 " + str(np.mean(confusionMatrix[0][0])) + "  " + str(np.mean(confusionMatrix[0][1])) + "\n" \
			"Pred=1 " + str(np.mean(confusionMatrix[1][0])) + "   " + str(np.mean(confusionMatrix[1][1]))
	print "confusionMatrix of Std: "
	print "      Expect=0 Expect=1\n" \
		  "Pred=0 " + str(np.std(confusionMatrix[0][0])) + "  " + str(np.std(confusionMatrix[0][1])) + "\n" \
			"Pred=1 " + str(np.std(confusionMatrix[1][0])) + "   " + str(np.std(confusionMatrix[1][1]))
	print "Accuracy Mean: " + str(np.mean(accuracy))
	print "Accuracy Std: " + str(np.std(accuracy))
	print "Precision in class 0 Mean:"+str(np.mean(precision0))
	print "Precision in class 0 Std:" + str(np.std(precision0))
	print "Precision in class 1 Mean:"+str(np.mean(precision1))
	print "Precision in class 1 Std:" + str(np.std(precision1))
	print "Recall in class 0 Mean:" + str(np.mean(recall0))
	print "Recall in class 0 Std:" + str(np.std(recall0))
	print "Recall in class 1 Mean:" + str(np.mean(recall1))
	print "Recall in class 1 Std:" + str(np.std(recall1))

"""
Show a graphic with the variation of accuracy with percent of data min to max step by step
divisors will be 1 if wants to go [1%...100%] or 0.01 if wants to go [0.1%...10%] if wants
to have diferents sections of graphic with diferent step ([0.01% to 1%] step by step and
[1% to 99%] with a 5 step) args min, max, step and divisor can be passed as a list
"""
def showGraphicOfAccuracyDimension(smooth =0, maxWordsInDict = None, limitIncluded = True,
								   min=[1], max=[100],step=[5], divisor=[1]):
	accuracy=[]
	percents=[]
	#Duck typing
	try:
		#If args are a list
		for i in range(len(divisor)):
			for j in range(min[i],max[i],step[i]):
				percents.append(j/float(divisor[i]))
				confusionMatrix = haveBasicCrossValidation(trainPercent=j/(float(divisor[i])*100), parseDates=False, smooth=smooth,
														   maxWordsInDict=maxWordsInDict, limitIncluded=limitIncluded)
				accuracy.append(em.getAccuracy(confusionMatrix=confusionMatrix))
	except:
		#if args are integers
		for j in range(min, max, step):
			percents.append(j / float(divisor))
			confusionMatrix = leaveOneOutValidation(percentOfData=j / float(divisor * 100), parseDates=False,
													   smooth=smooth)
			accuracy.append(em.getAccuracy(confusionMatrix=confusionMatrix))
	print percents
	print accuracy
	pl.xlabel("Percent of data")
	pl.ylabel("Accuracy")
	#Duck typing
	try:
		#if args are a lists
		pl.title("Evolution of Accuracy ["+str(min[0]/float(divisor[0]))+"% - "+str(max[-1]/float(divisor[-1]))+"%]")
	except:
		#if args are integers
		pl.title("Evolution of Accuracy [" + str(min / float(divisor)) + "% - " + str(max / float(divisor)) + "%]")
	pl.plot(percents, accuracy)
	pl.show()


def showGraficOfAccuracyDimensionOfDict(trainPercent = 0.33, smooth =0, limitIncluded = True,
								   minWords=[1], maxWords=None,step=[5]):
	assert 0<trainPercent<1
	accuracy=[]
	wordsInDict=[]
	#Duck typing
	try:
		#If args are a list
		for i in range(len(maxWords)):
			if maxWords[i] is None:
				data = bn.readDatabase(shuffle=True)
				data, _ = dt.splitDataInProportion(dataBase=data, trainPercent=trainPercent)
				bayesNet = bn.BayesianNetwork(dataBase=data)
				maxWords[i] = min(len(bayesNet.positiveTweetDict), len(bayesNet.negativeTweetDict))
			for j in range(minWords[i],maxWords[i],step[i]):
				wordsInDict.append(j)
				confusionMatrix = haveBasicCrossValidation(trainPercent=trainPercent, parseDates=False, smooth=smooth,
														   maxWordsInDict=j, limitIncluded=limitIncluded)
				accuracy.append(em.getAccuracy(confusionMatrix=confusionMatrix))
	except:
		#if args are integers
		if maxWords is None:
			data = bn.readDatabase(shuffle=True)
			data, _ = dt.splitDataInProportion(dataBase=data, trainPercent=trainPercent)
			bayesNet = bn.BayesianNetwork(dataBase=data)
			maxWords = min(len(bayesNet.positiveTweetDict), len(bayesNet.negativeTweetDict))
		for j in range(minWords, maxWords, step):
			wordsInDict.append(j)
			confusionMatrix = haveBasicCrossValidation(trainPercent=trainPercent, parseDates=False,
													   smooth=smooth,
													   maxWordsInDict=j, limitIncluded=limitIncluded)
			accuracy.append(em.getAccuracy(confusionMatrix=confusionMatrix))

	pl.xlabel("Words in dict")
	pl.ylabel("Accuracy")
	print wordsInDict
	print accuracy
	#Duck typing
	try:
		#if args are a lists
		pl.title("Evolution of Accuracy ["+str(minWords[0])+ " words - "+str(maxWords[-1])+" words]")
	except:
		#if args are integers
		pl.title("Evolution of Accuracy [" + str(minWords) + " words - " + str(maxWords) + " words]")
	pl.plot(wordsInDict, accuracy)
	pl.show()
