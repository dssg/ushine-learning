import logging
import pdb
import nltk
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import word_tokenize
from .junutils import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

from sklearn.svm import LinearSVC
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import scipy.stats as stats
from sklearn.cross_validation import StratifiedKFold
import sklearn.metrics as metrics
import scipy as sp


def getWords(text):
    return word_tokenize(text)
#  words = map(lambda x: x.strip(), text.split(' '))
#  words = filter(lambda x: len(x)>0, words)
#  return words


def bagOfWords(words):
    return dict([(word.lower(), True) for word in words])


def bagOfWordsNotInSet(words, badwords):
    return bagOfWords(set(words) - set(badwords))


def bagOfWordsExceptStopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bagOfWordsNotInSet(words, badwords)


class TooSkewedLabelsException(Exception):

    def __init__(self):
        pass
    pass


def dssgRunCV(binaryClassifierTrainer, dsetBinary, dssgVectorizerGenerator, nFolds=5):
    """
    raise TooSkewedLabelsException when CV can't be done

    """

    #--- Get staritified folds
    y = map(lambda x: 1 if x[1] == 'pos' else 0, dsetBinary)

    #- exception: if there is num(y==1)< 2 or num(y==0)<2, can't do CV.
    ySum = sum(y)
    if ySum < 2 or len(y) - ySum < 2:
        raise TooSkewedLabelsException()

    folds = StratifiedKFold(y, nFolds)

    #--- Compute accuracy
    accList = []
    precList = []
    recallList = []
    f1List = []
    allTruey = [x[1] for x in dsetBinary]
    allPredy = [float('nan')] * len(dsetBinary)
    for trainIdx, testIdx in folds:
        train = [dsetBinary[i] for i in trainIdx]
        test = [dsetBinary[i] for i in testIdx]

        #-- train
        binaryClassifier = binaryClassifierTrainer(train)

        #-- compute accuracy (or other stats?)
        predy = []
        for x in test:
            predScore = binaryClassifier.predictScore(x[0])
            predy.append('pos' if predScore['pos'] >= 0 else 'neg')

        truey = [x[1] for x in test]
        acc = metrics.accuracy_score(truey, predy)
        accList.append(acc)

        #-
        for i in range(len(test)):
            allPredy[testIdx[i]] = predy[i]

    #- compute prec, recall, f1
    prec = metrics.precision_score(allTruey, allPredy, pos_label='pos')
    recall = metrics.recall_score(allTruey, allPredy, pos_label='pos')
    f1 = metrics.f1_score(allTruey, allPredy, pos_label='pos')
    nTP = sum([allTruey[i] == 'pos' and allPredy[
              i] == 'pos' for i in range(len(allTruey))])
    nFP = sum([allTruey[i] == 'neg' and allPredy[
              i] == 'pos' for i in range(len(allTruey))])
    nTN = sum([allTruey[i] == 'neg' and allPredy[
              i] == 'neg' for i in range(len(allTruey))])
    nFN = sum([allTruey[i] == 'pos' and allPredy[
              i] == 'neg' for i in range(len(allTruey))])

    resDic = {}
    resDic['nFolds'] = nFolds
    me = np.mean(accList)
    resDic['mean of accuracy'] = me
    stderr = stats.sem(accList)
    resDic['standard error of accuracy'] = stderr
    alpha = 0.05
    # failure rate
    dev = stderr * stats.t.ppf(1 - alpha / 2, len(accList) - 1)
    resDic['95% confidence interval'] = (me - dev, me + dev)
    resDic['precision'] = prec
    resDic['recall'] = recall
    resDic['f1'] = f1
    resDic['nTP'] = nTP
    resDic['nFP'] = nFP
    resDic['nTN'] = nTN
    resDic['nFN'] = nFN

    return resDic
#
# UnigramExtractor
#


class DssgUnigramExtractor(object):

    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()
        self._stopwordSet = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()

    def extractUnigrams(self, text):
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace(
            '&quot;', '"').replace('&amp;', '&').replace('&nbsp;', ' ')
        text = nltk.clean_html(text)
        tokens = self._tokenizer.tokenize(text)

        newTokens = []
        for tok in tokens:
            #- lowercase, remove '
            tok = tok.lower().strip("`'.,-_*/:;\\!@#$%^&*()=\"")

            #- remove stopwords, one character word, only numbers
            #- remove one character word
            #- remove only numbers
            if (tok in self._stopwordSet or len(tok) <= 1 or isAllNumbers(tok)):
                continue

            #- apply stemming
# oldTok = copy.deepcopy(tok) # for debug
            tok = self._stemmer.stem(tok)
            if (tok in self._stopwordSet):  # sometimes a token is like 'theres' and becomes stopword after stemming
                continue

            newTokens.append(tok)
        return newTokens

#
# Vectorizer
#


class DssgVectorizer(object):

    def __init__(self):
        pass

    def fitTransform(self, messageList):
        #- find unigram, compute length, POS tagging and so on...
        raise NotImplementedError("")

    def transform(self, messageList):
        raise NotImplementedError("")

    def getFeatureNameList(self):
        raise NotImplementedError("")

    pass


class DssgVectorizerUnigramCount(DssgVectorizer):

    def __init__(self, unigramExtractor, minFreq=1):
        self._unigramExtractor = unigramExtractor
        self._minFreq = minFreq
        self._unigramCountDic = {}
        self._unigramIdxDic = {}

    def fitTransform(self, messageList):
        assert(len(self._unigramIdxDic) == 0)

        #- find unigrams
        unigramCountDic = {}
        cache = []
        for msg in messageList:
            unigramList = self._unigramExtractor.extractUnigrams(
                msg['title'] + ' ' + msg['description'])
#            unigramList = self._extractUnigramFunc( msg['description'])
#            unigramList = self._extractUnigramFunc( msg['title'])
            cache.append(unigramList)
            for unigram in unigramList:
                if (unigram not in unigramCountDic):
                    unigramCountDic[unigram] = 0
                unigramCountDic[unigram] += 1

        #- cutoff
        pairs = unigramCountDic.items()
        pairs = filter(lambda x: x[1] >= self._minFreq, pairs)

        unigramCountDic = dict(pairs)
        unigramList = sorted(unigramCountDic.keys())
        self._unigramIdxDic = dict(zip(unigramList, range(len(unigramList))))
        self._unigramCountDic = unigramCountDic

        #- transform
        return self.transform(messageList, cache=cache)

    def transform(self, messageList, cache=None):
        nUnigram = len(self._unigramIdxDic)
        rowNum = 0
        iList = []
        jList = []
        valueList = []

        for i in range(len(messageList)):
            msg = messageList[i][
                'title'] + ' ' + messageList[i]['description']
#            msg = messageList[i]['description']
#            msg = messageList[i]['title']
            row = {}
            unigramList = cache[i] if cache is not None else \
                self._unigramExtractor.extractUnigrams(msg)

            for unigram in unigramList:
                if (unigram in self._unigramIdxDic):
                    if (unigram not in row):
                        row[unigram] = 1
                    else:
                        row[unigram] += 1
            for (k, v) in row.iteritems():
                iList.append(rowNum)
                jList.append(self._unigramIdxDic[k])
                valueList.append(v)

            rowNum += 1

        return coo_matrix((valueList, (iList, jList)), shape=(len(messageList), len(self._unigramIdxDic)))

    def getFeatureNameList(self):
        return self._unigramIdxDic.keys()

    pass


class DssgVectorizerUnigramBySklearn(DssgVectorizer):

    def __init__(self, minFreq=1):
        self._vectorizer = None
        pass

    def fitTransform(self, messageList):
        assert(self._vectorizer is None)
        self._vectorizer = CountVectorizer(
            min_df=1, stop_words='english', token_pattern=u'(?u)\\b\\w*[a-zA-Z]\\w*\\b')
        return self._vectorizer.fit_transform([msg['title'] + ' ' + msg['description'] for msg in messageList])

    def transform(self, messageList):
        return self._vectorizer.transform([msg['title'] + ' ' + msg['description'] for msg in messageList])

    def getFeatureNameList(self):
        return self._vectorizer.vocabulary_.keys()

    pass


#
# Binary Classifiers
#

class DssgBinaryClassifier(object):

    """ This is meant to be an abstract class
    """
    def __init__(self):
        pass

    @staticmethod
    def train(binaryLabeledMessageList):
        """ labels should be 'pos'/'neg'
        """
        raise NotImplementedError("")

    def predictScore(message):
        """ returns a score dictionary that looks like: {"pos":0.8, "neg":0.2}
        """
        raise NotImplementedError("")

    pass


class DssgBinaryClassifierMajorityVote(DssgBinaryClassifier):

    """
    Majority vote classifier.
    """
    def __init__(self, posNegPrbDic):
        self._posNegPrbDic = posNegPrbDic

    @staticmethod
    def train(binaryLabeledMessageList):
        nPos = countTruth(lambda x: x[1] == 'pos', binaryLabeledMessageList)
        nNeg = countTruth(lambda x: x[1] == 'neg', binaryLabeledMessageList)
        assert(nPos + nNeg == len(binaryLabeledMessageList))
        posNegPrbDic = {'pos': float(nPos) / (
            nPos + nNeg), 'neg': float(nNeg) / (nPos + nNeg)}

        dssgClassifier = DssgBinaryClassifierMajorityVote(posNegPrbDic)
        return dssgClassifier

    def predictScore(self, message):
        """
        This turns probability into score in range [-.5, 5]
        """
        scoreDic = {}
        scoreDic['pos'] = (_posNegPrbDic['pos'] - .5)
        scoreDic['neg'] = (_posNegPrbDic['neg'] - .5)
        return _posNegPrbDic

    pass


class DssgBinaryClassifierNaiveBayes(DssgBinaryClassifier):

    def __init__(self, nbClassifier):
        self._nbClassifier = nbClassifier
        pass

    @classmethod
    def train(cls, binaryLabeledMessageList):
        tmp = map(lambda x: (cls._getFeatures(x[
                  0]), x[1]), binaryLabeledMessageList)
        nbClassifier = NaiveBayesClassifier.train(tmp)
        dssgClassifier = DssgBinaryClassifierNaiveBayes(nbClassifier)
        return dssgClassifier

    def predictScore(self, message):
        probs = self._nbClassifier.prob_classify(self._getFeature(message))
        return {'pos': probs.prob('pos'), 'neg': probs.prob('neg')}

    @staticmethod
    def _getFeatures(text):
        return bagOfWordsExceptStopwords(getWords(text.lower()))

    pass


class DssgBinaryClassifierSVC(DssgBinaryClassifier):

    def __init__(self, classifier, dssgVectorizer):
        assert(isinstance(dssgVectorizer, DssgVectorizer))
        self._classifier = classifier
        self._dssgVectorizer = dssgVectorizer

    @classmethod
    def train(cls, binaryLabeledMessageList, dssgVectorizer, balance=False):
        msgList = map(lambda x: x[0], binaryLabeledMessageList)
        y = np.array(map(lambda x: 1 if x[
                     1] == 'pos' else 0, binaryLabeledMessageList))
        X = dssgVectorizer.fitTransform(msgList)

        if (balance is True):
            classifier = LinearSVC(
                loss='l2', penalty='l1', dual=False, tol=1e-3,
                random_state=0, class_weight='auto')
        else:
            classifier = LinearSVC(
                loss='l2', penalty='l1', dual=False, tol=1e-3,
                random_state=0)

        classifier.fit(X, y)

        dssgClassifier = cls(classifier, dssgVectorizer)
        return dssgClassifier

    def predictScore(self, message):
        featVec = self._dssgVectorizer.transform([message])
        df = self._classifier.decision_function(featVec)
        scoreDic = {}
        scoreDic['pos'] = df[0]
        scoreDic['neg'] = -df[0]
        return scoreDic

    pass

#
# Category Classifiers
#


class DssgCategoryClassifier(object):
    _binaryClassifierTrainer = None
    _classifierDic = {}
    _categoryList = []
    _trainStats = {}

    def getTrainStats(self):
        return self._trainStats

    def __init__(self, binaryClassifierTrainer, classifierDic, categoryList, trainStats):
        self._binaryClassifierTrainer = binaryClassifierTrainer
        self._classifierDic = classifierDic
        self._categoryList = categoryList
        self._trainStats = trainStats

    @classmethod
    def train(cls, binaryClassifierTrainer, messageList, dssgVectorizerGenerator=None):
        """ A constructor for a classifier.
        messageList is a list of message in dictionary. Major keys are 'title', 'description', and 'categories', and so on.
        """
        #- default vectorizer
        if (dssgVectorizerGenerator is None):
            dssgVectorizerGenerator = lambda: DssgVectorizerUnigramBySklearn()
        #--- Empty variables
        classifierDic = {}
        categoryList = []

        #--- extract category list
        categorySet = set()
        for msg in messageList:
            categorySet.update(msg['categories'])
        categoryList = sorted(list(categorySet))

        #--- for each category, train a binary classifier
        classifierDic = {}
        trainStats = {}
        totNTP = 0
        totNFP = 0
        totNTN = 0
        totNFN = 0
        for cat in categoryList:
            logging.info('Category: %s', cat)
            dset = []
            for msg in messageList:
                categoryTitleList = msg['categories']
                binaryLabel = 'pos' if cat in categoryTitleList else 'neg'
                dset.append((msg, binaryLabel))

            nPos = countTruth(lambda x: x[1] == 'pos', dset)
            nNeg = countTruth(lambda x: x[1] == 'neg', dset)
            assert(len(dset) == nPos + nNeg)

            v = float(nPos) / (nPos + nNeg)
            mvAcc = max([v, 1 - v])
            logging.info(
                '  nPos,nNeg = (%d / %d), majority vote acc = %.3f', nPos, nNeg, mvAcc)

            statDic = {'nPos': nPos, 'nNeg': nNeg, 'majority vote acc': mvAcc}

            #--- run cross validation to estimate performance
            try:
                cvResDic = dssgRunCV(
                    binaryClassifierTrainer, dset, dssgVectorizerGenerator)
                strList = ('nFolds', 'mean of accuracy',
                           '95% confidence interval', 'precision', 'recall', 'f1')
                msg = '%2d-CV: %s = %.3f, %s = [%.3f, %.3f]\n' % (cvResDic[strList[0]], strList[
                                                                  1], cvResDic[strList[1]], strList[2], cvResDic[strList[2]][0], cvResDic[strList[2]][1])
                for i in [1, 3, 4, 5]:
                    msg += '       %s = %.3f' % (
                        strList[i], cvResDic[strList[i]])
                    if (i != 5):
                        msg += '\n'

            except TooSkewedLabelsException:
                cvResDic = {}
                msg = 'CV accuracy estimate is not available due to skewed labels'

            statDic['CV stat'] = cvResDic
            trainStats[cat] = statDic
            logging.info('  ' + msg)

            #--- build a classifier using all data.
            if (nPos == 0 or nNeg == 0):
                #- when only one label presents, do majority vote..
                dssgClassifier = DssgBinaryClassifierMajorityVote.train(dset)
                pass

            dssgClassifier = binaryClassifierTrainer(dset)
            nVoca = len(dssgClassifier._dssgVectorizer.getFeatureNameList())
            logging.info('  nVoca = %d', nVoca)

            # add classifier to classifier dictionary.
            classifierDic[cat] = dssgClassifier

            logging.info('  type = %s', str(type(dssgClassifier)))
            totNTP += statDic['CV stat']['nTP']
            totNFP += statDic['CV stat']['nFP']
            totNTN += statDic['CV stat']['nTN']
            totNFN += statDic['CV stat']['nFN']

        meanF1 = np.mean(np.array(map(lambda x: x[
                         'CV stat']['f1'], trainStats.values())))
        logging.info('Overall mean F1 = %.3f', meanF1)

        totPrecision = float(totNTP) / (totNTP + totNFP)
        totRecall = float(totNTP) / (totNTP + totNFN)
        logging.info('Overall aggregate prec = %.3f, recall = %.3f, F1 = %.3f',
                     totPrecision, totRecall, sp.stats.hmean([totPrecision, totRecall]))
        return cls(binaryClassifierTrainer, classifierDic, categoryList, trainStats)

    def predictScore(self, message):
        """
        Predicts labels in a form of {'label1':0.9, 'label2':0.8, ...}
        """

        labelProbDic = {}
        for cat in self._categoryList:
            probs = self._classifierDic[cat].predictScore(message)
            labelProbDic[cat] = probs['pos']

        return labelProbDic

    pass  # end of class
