import BayesianNetwork as bn
import ErrorMeasurements as em
import Validators as val

#List with most common words that appears aproximately equals in positive and negative tweets (noise)
mostCommonWords = ['i', 'to', 'the', 'you', 'and', 'a', 'on', 'is', 'it', 'in', 'for', 'me', 'that', 'with', 'of', 'my']
if __name__ == "__main__":
	print "Starting..."
	dataFrame = bn.readDatabase(shuffle=True)
	#data,_ = dt.splitDataInProportion(dataBase=dataFrame, trainPercent=0.33)
	#val.CrossValidationMedian(executions=10,trainPercent=0.66, parseDates=False, smooth=0)
	print em.printAllParams(val.haveKFoldValidation(data=dataFrame, k=5, smooth=0))
	print em.getF1Score(val.leaveOneOutValidation(data = dataFrame, smooth=0, filter=[]),positiveClass=0)
	print em.getF1Score(val.leaveOneOutValidation(data=dataFrame, smooth=1, filter=[]), positiveClass=0)