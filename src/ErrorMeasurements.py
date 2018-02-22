#Predicted on rows
#expected on columns

def printMatrix(confusionMatrix):
    print "confusionMatrix: "
    print "      Expect=0 Expect=1\n" \
          "Pred=0 "+str(confusionMatrix[0][0])+"  "+str(confusionMatrix[0][1])+"\n" \
          "Pred=1 "+str(confusionMatrix[1][0])+"   "+str(confusionMatrix[1][1])

def printAllParams(confusionMatrix, positiveClass=1):
    otherClass = 0 if positiveClass == 1 else 1
    print "Confusion Matrix:"
    printMatrix(confusionMatrix=confusionMatrix)
    print "Accuracy: "+str(getAccuracy(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "Error (1-Accuracy): " + str(1-getAccuracy(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "F1-Score: " + str(getF1Score(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "Precision in class "+str(positiveClass)+": " + str(
        getPrecision(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "Precision in class " + str(otherClass) + ": " + str(
        getPrecision(confusionMatrix=confusionMatrix, positiveClass=otherClass))
    print "Recall in class "+str(positiveClass)+": "+ str(
        getRecall(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "Recall in class " + str(otherClass) + ": " + str(
        getRecall(confusionMatrix=confusionMatrix, positiveClass=otherClass))
    print "Specificity (True Negative Rate)  in class " + str(positiveClass) +": "+ str(
        getSpecifitiy(confusionMatrix=confusionMatrix, positiveClass=positiveClass))
    print "Specificity (True Negative Rate)  in class " + str(otherClass) + ": " + str(
        getSpecifitiy(confusionMatrix=confusionMatrix, positiveClass=otherClass))


def analyzeMatrix(confusionMatrix,positiveClass=1):
    negativeClass = 0 if positiveClass == 1 else 1
    TP = confusionMatrix[positiveClass][positiveClass]
    TN = confusionMatrix[negativeClass][negativeClass]
    FP = confusionMatrix[negativeClass][positiveClass]
    FN = confusionMatrix[positiveClass][negativeClass]
    return TP, TN, FP, FN

def getAccuracy(confusionMatrix, positiveClass=1):
   TP,TN,FP,FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
   return float(TP+TN)/float(TP+FP+FN+TN)

def getF1Score(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(2*TP)/float(2*TP+FP+FN)

def getRecall(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(TP)/float(TP+FN)

def getPrecision(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(TP)/float(TP+FP)

def getSpecifitiy(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(TN)/float(TN+FP)

def getFalseNegativeRate(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(FN)/float(TP+FN)

def getFalseDiscoveryRate(confusionMatrix, positiveClass=1):
    TP, TN, FP, FN = analyzeMatrix(confusionMatrix=confusionMatrix, positiveClass=positiveClass)
    return float(FP)/float(TP+FP)