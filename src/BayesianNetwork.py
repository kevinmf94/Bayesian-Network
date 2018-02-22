import numpy as np
import pandas
import math
import time

positiveLabel = 1
negativeLabel = 0
class BayesianNetwork:
    _epsilon = 0.000001
    _epsilonLog = math.log(_epsilon)
    confusionMatrix = [[0,0],[0,0]]
    positiveTweetDict = {} #dict of one key [word] : count of this word in positive examples #OK
    negativeTweetDict= {} #dict of one key [word] : count of this word in negative examples #OK
    probOfWord = {} #dict of one key [word] : prob of word
    totalTweetsInDate = {} #dict of one key [date] : counts of tweets in date
    probOfTweetInDate = {} # dict of one key [date] : prob of tweet in date
    joinDatexTweet = {} #dict of double key [date][word] : countOfWordsInDate
    positiveWordsCount = 0 #total number of positive words #OK
    negativeWordsCount = 0 #total number of negative words #OK
    totalWords = 0 #total number of positive words
    probPositiveWord = 0.0 #total prob of have a positive word
    probNegativeWord = 0.0 #total prob of have a negative word
    data = None #database
    # definition of Bayesian net Graph (deprecated)
    dependencies = {'tweetDate':[], 'tweetText':['tweetDate'], 'sentimentLabel':['tweetText','tweetDate']}

    """
    Constructor of the object BayesianNetwork makes a Naive Bayes network filling his associated data
    """
    def __init__(self, dataBase = None, databaseName="../FinalStemmedSentimentAnalysisDataset.csv", parseDates = False,
                 maxWordsInDict = None, limitIncludeInWordsDict = True, confusionMatrix=[[0,0],[0,0]]):
        timeOfRead = 0
        timeStart = time.clock()
        if dataBase is None:
            self.data = readDatabase(databaseName= databaseName, parseDates=parseDates)
            timeOfRead = time.clock() - timeStart
            print "CSV Readed in: " + str(timeOfRead) + " Seconds"
        else:
            self.data = dataBase
        self.confusionMatrix=confusionMatrix
        self.positiveTweetDict, self.negativeTweetDict, self.positiveWordsCount, self.negativeWordsCount, uniqueWordsSet = \
            createPositiveAndNegativeCountDicts(data=self.data, maxWordsInDict=maxWordsInDict, limitIncludeInWordsDict=limitIncludeInWordsDict)
        timeOfClassifing = time.clock() - timeOfRead - timeStart
        print "Dictionaris created in: " + str(timeOfClassifing) + " Seconds"


        #filling totalDictionarys
        self.totalWords = self.positiveWordsCount+self.negativeWordsCount
        #self.totalTweetsInDate = dict(self.data.tweetDate.value_counts())
        self.probPositiveTweet = float(self.positiveWordsCount)/float(self.totalWords)
        self.probNegativeTweet = float(self.negativeWordsCount)/float(self.totalWords)
        """
        #this code is @deprecated, because joinDistributions are irrelevant in that problem (dataAttribute only introduce noise
        #filling probOfWord
        for word in uniqueWordsSet:
            self.probOfWord[word] = float(self.totalOcurrencesOfWord(word))/float(self.totalWords)
        timeOfTotalWordProbs = time.clock() - timeOfClassifing - timeOfRead - timeStart
        print "Prob of word generated in: " + str(timeOfTotalWordProbs) + " Seconds"

        #filling probOfTweetInDate
        tweetDatesShape = float(self.data.tweetDate.shape[0])
        for date in self.totalTweetsInDate.keys():
            self.probOfTweetInDate[date] = float(self.totalTweetsInDate[date])/tweetDatesShape
        timeOfDateProbsFilled = time.clock() - timeOfTotalWordProbs - timeOfClassifing - timeOfRead - timeStart
        print "Prob of have tweet in date filled in: " + str(timeOfDateProbsFilled) + " Seconds"

        #filling Join table between dates and words #THIS IS PARALELIZABLE
        for date in self.totalTweetsInDate.keys():
            self.joinDatexTweet[date] =  createGeneralCountDict(self.data[self.data['tweetDate'] == date])
        timeOfJoinTableFilled = time.clock() - timeOfDateProbsFilled - timeOfTotalWordProbs - timeOfClassifing - timeOfRead - timeStart
        print "Join table of say's a word in date filled in: " + str(timeOfJoinTableFilled) + " Seconds"
        """

    """
    Gets the total ocurrences of a word and adds the smothing offset if is applied
    """
    def totalOcurrencesOfWord(self, word, smoothing = 0):
        ocurrences = 0
        if self.positiveTweetDict.has_key(word):
            ocurrences+=self.positiveTweetDict[word]
        if self.negativeTweetDict.has_key(word):
            ocurrences+=self.negativeTweetDict[word]
        ocurrences+=smoothing
        return ocurrences

    """
    Have a validation of a set of data, and returns the error aplying Laplace Smoothing
    """
    def haveSmoothValidation(self, tweetTexts, tweetClasses, smooth=1):
        positiveSmoothDivisor = len(self.positiveTweetDict) * smooth
        negativeSmoothDivisor = len(self.negativeTweetDict) * smooth
        for i in range(tweetTexts.shape[0]):
            probPositive = 0.0
            probNegative = 0.0
            for word in tweetTexts[i].lower().split():
                negativeWords = (self.negativeTweetDict[word] + smooth) if self.negativeTweetDict.has_key(
                    word) else smooth
                positiveWords = (self.positiveTweetDict[word] + smooth) if self.positiveTweetDict.has_key(
                    word) else smooth
                probPositive += math.log(positiveWords / (self.positiveWordsCount + positiveSmoothDivisor))
                probNegative += math.log(negativeWords / (self.negativeWordsCount + negativeSmoothDivisor))
            probPositive += math.log(self.probPositiveTweet)
            probNegative += math.log(self.probNegativeTweet)
            if probPositive > probNegative:
                self.FillConfusionMatrix(predicted=positiveLabel, expected=tweetClasses[i])
            else:
                self.FillConfusionMatrix(predicted=negativeLabel, expected=tweetClasses[i])
        return self.confusionMatrix

    """
    Have a validation of a set of data, and returns the confusionMatrix without laplace Smoothing
    """
    def haveNonSmoothValidation(self, tweetTexts, tweetClasses):
        for i in range(tweetTexts.shape[0]):
            probPositive = 0.0
            probNegative = 0.0
            for word in tweetTexts[i].lower().split():
                if self.positiveTweetDict.has_key(word):
                    probPositive += math.log(self.positiveTweetDict[word] / self.positiveWordsCount)
                else:
                    probPositive += self._epsilonLog
                if self.negativeTweetDict.has_key(word):
                    probNegative += math.log(self.negativeTweetDict[word] / self.negativeWordsCount)
                else:
                    probNegative += self._epsilonLog
            probPositive += math.log(self.probPositiveTweet)
            probNegative += math.log(self.probNegativeTweet)
            if probPositive > probNegative:
                self.FillConfusionMatrix(predicted=positiveLabel, expected=tweetClasses[i])
            else:
                self.FillConfusionMatrix(predicted=negativeLabel, expected=tweetClasses[i])
        return self.confusionMatrix

    """
    Have a validation of a set of data, and returns the error
    """
    def haveValidation(self, validationSet, smooth = 0):
        vectorTweetText = validationSet['tweetText'].values
        vectorSentLabel = validationSet['sentimentLabel'].values
        if smooth>0 and smooth<=1:
            return self.haveSmoothValidation(tweetTexts=vectorTweetText, tweetClasses=vectorSentLabel, smooth=smooth)
        else:
            return self.haveNonSmoothValidation(tweetTexts=vectorTweetText, tweetClasses=vectorSentLabel)

    """
    THIS FUNCTION IS ONLY VALID FOR LEAVE ONE OUT METHOD. gets the label associated to a Tweet without aplying LaplaceSmoothing
    """
    def getLabelOfTweetNonSmooth(self, tweet):
        probPositive = 0.0
        probNegative = 0.0
        for word in tweet:
            if self.positiveTweetDict.has_key(word) and self.positiveTweetDict[word] != 0:
                probPositive += math.log(self.positiveTweetDict[word] / self.positiveWordsCount)
            else:
                probPositive += self._epsilonLog
            if self.negativeTweetDict.has_key(word) and self.negativeTweetDict[word] != 0:
                probNegative += math.log(self.negativeTweetDict[word] / self.negativeWordsCount)
            else:
                probNegative += self._epsilonLog
        probPositive += math.log(self.probPositiveTweet)
        probNegative += math.log(self.probNegativeTweet)
        if probPositive > probNegative:
            return positiveLabel
        else:
            return negativeLabel

    """
    THIS FUNCTION IS ONLY VALID FOR LEAVE ONE OUT METHOD. gets the label associated to a Tweet aplying LaplaceSmoothing
    """
    def getLabelOfTweetWithSmooth(self, tweet, smooth):
        probPositive = 0.0
        probNegative = 0.0
        positiveSmoothDivisor = len(self.positiveTweetDict) * smooth
        negativeSmoothDivisor = len(self.negativeTweetDict) * smooth
        for word in tweet:
            negativeWords = self.negativeTweetDict[word] + smooth if self.negativeTweetDict.has_key(
                word) else smooth
            positiveWords = (self.positiveTweetDict[word] + smooth) if self.positiveTweetDict.has_key(
                word) else smooth
            probPositive += math.log(positiveWords / (self.positiveWordsCount + positiveSmoothDivisor))
            negativeWords = (self.negativeTweetDict[word] + smooth) if self.negativeTweetDict.has_key(
                word) else smooth
            probNegative += math.log(negativeWords / (self.negativeWordsCount + negativeSmoothDivisor))
        probPositive += math.log(self.probPositiveTweet)
        probNegative += math.log(self.probNegativeTweet)
        if probPositive > probNegative:
            return positiveLabel
        else:
            return negativeLabel

    """
    Returns the label of a string. If you wants to use smooth, put smooth variable in range (0,1]. This function have
    a very low performance. Preferible used only for debug tests
    """
    def getLabelOfTweet(self, tweet, smooth=0):
        probPositive = 0.0
        probNegative = 0.0
        smoothFlag = False
        if smooth>0 and smooth<=1:
            smoothFlag = True
            positiveSmoothDivisor = len(self.positiveTweetDict)*smooth
            negativeSmoothDivisor = len(self.negativeTweetDict)*smooth
        for word in tweet.lower().split():
            negativeWords = self.negativeTweetDict[word]+smooth if self.negativeTweetDict.has_key(word) else smooth
            if smoothFlag:
                positiveWords = (self.positiveTweetDict[word] + smooth) if self.positiveTweetDict.has_key(word) else smooth
                probPositive += math.log(positiveWords/(self.positiveWordsCount+positiveSmoothDivisor))
            else:
                if self.positiveTweetDict.has_key(word):
                    probPositive += math.log(self.positiveTweetDict[word]/self.positiveWordsCount)
                else:
                    probPositive += self._epsilonLog
            if smoothFlag:
                negativeWords = (self.negativeTweetDict[word] + smooth) if self.negativeTweetDict.has_key(word) else smooth
                probNegative += math.log(negativeWords/(self.negativeWordsCount+negativeSmoothDivisor))
            else:
                if self.negativeTweetDict.has_key(word):
                    probNegative += math.log(self.negativeTweetDict[word]/self.negativeWordsCount)
                else:
                    probNegative += self._epsilonLog

        probPositive += math.log(self.probPositiveTweet)
        probNegative += math.log(self.probNegativeTweet)
        if probPositive>probNegative:
            return positiveLabel
        else:
            return negativeLabel

    """
    This function is optimized for have a leaveOneOut validation, all functions that it uses are optimized for it and
    can't be used for others purposes. Returns the error of LeaveOneOutValidation
    """
    def haveLeaveOneOutValidation(self, smooth=0):
        if smooth == 0:
            for i in range(self.data.shape[0]):
                tweetErased, label = self.eraseFromDictionariesOneTweet(index=i)
                prediction = self.getLabelOfTweetNonSmooth(tweet=tweetErased)
                self.FillConfusionMatrix(predicted=prediction, expected=label)
                self.undoLeaveOneOutErase(tweetErased=tweetErased, label=label)
        else:
            for i in range(self.data.shape[0]):
                tweetErased, label = self.eraseFromDictionariesOneTweet(index=i)
                prediction = predicted=self.getLabelOfTweetWithSmooth(tweet=tweetErased, smooth=smooth)
                self.FillConfusionMatrix(predicted=prediction, expected=label)
                self.undoLeaveOneOutErase(tweetErased=tweetErased, label=label)
        return self.confusionMatrix

    def FillConfusionMatrix(self, predicted, expected):
        if predicted == positiveLabel:
            if expected == negativeLabel:
                self.confusionMatrix[positiveLabel][negativeLabel] += 1
            else:
                self.confusionMatrix[positiveLabel][positiveLabel] += 1
        else:
            # is negative
            if expected == positiveLabel:
                self.confusionMatrix[negativeLabel][positiveLabel] += 1
            else:
                self.confusionMatrix[negativeLabel][negativeLabel] += 1


    """
    Undo the work of eraseFromDictionariesOneTweet() function, WARNING: for optimization, this function don't update
    the value of probOfTweetX. cause this will be recalculated on the next iteration
    """
    def undoLeaveOneOutErase(self, tweetErased, label):
        self.totalWords += len(tweetErased)
        if label == 0:
            self.negativeWordsCount += len(tweetErased)
            for word in tweetErased:
                self.negativeTweetDict[word] += 1
        else:
            self.positiveWordsCount += len(tweetErased)
            for word in tweetErased:
                self.positiveTweetDict[word] += 1

    """
    Erases a tweet and returns a tuple with (The splitted Tweet Erased, sentimentLabel)
    """
    def eraseFromDictionariesOneTweet(self, index):
        tweetToErase = self.data['tweetText'][index].split()
        label = self.data['sentimentLabel'][index]
        self.totalWords -= len(tweetToErase)
        if label == 0:
            self.negativeWordsCount -= len(tweetToErase)
            self.probNegativeTweet = float(self.negativeWordsCount) / float(self.totalWords)
            for word in tweetToErase:
                self.negativeTweetDict[word] -= 1
        else:
            self.positiveWordsCount -= len(tweetToErase)
            self.probPositiveTweet = float(self.positiveWordsCount) / float(self.totalWords)
            for word in tweetToErase:
                self.positiveTweetDict[word] -= 1
        return tweetToErase, label
    """
    puts to zero the confusion matrix
    """
    def restartConfusionMatrix(self):
        self.confusionMatrix = [[0,0],[0,0]]


"""
Gets the database passed by url as pandas array, if filter is specified erase the words that appears in the filter
"""
def readDatabase(databaseName="../FinalStemmedSentimentAnalysisDataset.csv", parseDates=False, shuffle = True, filter = []):
    if parseDates:
        dateparser = lambda parser: pandas.datetime.strptime(parser, '%d/%m/%Y')
        data = pandas.read_csv(databaseName, sep=";", usecols=[1, 2, 3],
                               dtype={'tweetText': np.str, 'sentimentLabel': np.int8},
                               na_filter=False, parse_dates=['tweetDate'], date_parser=dateparser)
    else:
        data = pandas.read_csv(databaseName, sep=";", usecols=[1, 2, 3],
                               dtype={'tweetText': np.str, 'tweetDate': np.str, 'sentimentLabel': np.int8},
                               na_filter=False)
    if filter != []:
        timeStart = time.clock()
        tweets = data['tweetText'].str.split()
        tweetTextValues = data['tweetText'].values
        for i in range(len(tweets)):
            filteredTweet = [word for word in tweets[i] if word not in filter]
            if len(filteredTweet)<len(tweets[i]):
                tweetTextValues[i] = ' '.join(filteredTweet)
        timeOfFilter = time.clock()-timeStart
        print "Filter applied in "+str(timeOfFilter)+" seconds"
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    return data

"""
Makes a positive and negative dictionary with the number of ocurrences of each word in data
"""
def createPositiveAndNegativeCountDicts(data, maxWordsInDict = None, limitIncludeInWordsDict = True):

    negativeTweets = data['tweetText'][data['sentimentLabel'] == 0].str.split()
    positiveTweets = data['tweetText'][data['sentimentLabel'] == 1].str.split()
    # print data['tweetText'][632]
    negativeTweetDict = {}
    positiveTweetDict = {}
    negativeWordsCount = 0.0
    positiveWordsCount = 0.0
    for tweet in positiveTweets:
        for word in tweet:
            positiveWordsCount += 1
            if not positiveTweetDict.has_key(word):
                positiveTweetDict[word] = 1
            else:
                positiveTweetDict[word] += 1
    for tweet in negativeTweets:
        for word in tweet:
            negativeWordsCount += 1
            if not negativeTweetDict.has_key(word):
                negativeTweetDict[word] = 1
            else:
                negativeTweetDict[word] += 1
    if maxWordsInDict is None:
        uniqueWordsSet = set(positiveTweetDict.keys())
        uniqueWordsSet.union(set(negativeTweetDict.keys()))
        return positiveTweetDict, negativeTweetDict, positiveWordsCount, negativeWordsCount, uniqueWordsSet
    else:
        return truncateDictionaries(positiveTweetDict, negativeTweetDict, positiveWordsCount, negativeWordsCount,
                                    newLength=maxWordsInDict, limitIncluded=limitIncludeInWordsDict)

"""
Truncate a dictionary and updates de associated information. The new dictionaries will only contains the (approximately) newLength words
with more ocurrences in each of these. limitInclude flag will determine if the words with the same threshold that the
 limit will be included.
"""
def truncateDictionaries(positiveTweetDict, negativeTweetDict, positiveWordsCount, negativeWordsCount, newLength = None,
                         limitIncluded = True):
    #set the aux values
    positiveValues = np.array(positiveTweetDict.values())
    negativeValues= np.array(negativeTweetDict.values())
    positiveValues = np.sort(positiveValues, kind='heapsort')[::-1] #heap sort is the fastest method (O(n*log(n)))
    negativeValues = np.sort(negativeValues, kind='heapsort')[::-1] #heap sort is the fastest method (O(n*log(n)))
    positiveWordsErased = 0
    negativeWordsErased = 0
    positiveThreshold = positiveValues[min(newLength, len(positiveValues))]
    negativeThreshold = negativeValues[min(newLength, len(negativeValues))]

    if not limitIncluded:
        positiveThreshold+=1
        negativeThreshold+=1

    #truncate dicts
    for key in positiveTweetDict.keys():
        if positiveTweetDict[key] < positiveThreshold:
            positiveWordsErased += positiveTweetDict[key]
            del positiveTweetDict[key]
    for key in negativeTweetDict.keys():
        if negativeTweetDict[key] < negativeThreshold:
            negativeWordsErased += negativeTweetDict[key]
            del negativeTweetDict[key]

    #updateWordCounts
    positiveWordsCount -= positiveWordsErased
    negativeWordsCount -= negativeWordsErased

    #make WordSet
    uniqueWordsSet = set(positiveTweetDict.keys())
    uniqueWordsSet.union(set(negativeTweetDict.keys()))
    return positiveTweetDict, negativeTweetDict, positiveWordsCount, negativeWordsCount, uniqueWordsSet




"""
@deprecated
Makes a dictionary with the number of ocurrences of each word in data
"""
def createGeneralCountDict(data):
    tweets = data['tweetText'].str.lower().str.split()
    tweetDict = {}
    for tweet in tweets:
        for word in tweet:
            if not tweetDict.has_key(word):
                tweetDict[word] = 1
            else:
                tweetDict[word] += 1
    return tweetDict

