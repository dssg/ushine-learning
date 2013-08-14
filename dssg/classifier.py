from .util import * 
from .vectorizer import *
import platt 

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np;
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import scipy.stats as stats;
from sklearn.cross_validation import StratifiedKFold
import sklearn.metrics as metrics;
import scipy as sp;
import ipdb;
import pprint


class TooSkewedLabelsException(Exception):
    def __init__(self):
        pass
    pass


################################################################################
# Binary Classifiers
################################################################################

class DssgBinaryClassifier(object):
    """ This is meant to be an abstract class 
    """
    def __init__(self):
        pass

    @staticmethod
    def train(binaryLabeledMessageList):
        """ labels should be 'pos'/'neg'
        """
        raise NotImplementedError("");

    def predictScore(self, message):
        """ returns a score dictionary that looks like: {"pos":.2, "neg":-.2}
        """
        raise NotImplementedError("");

    def predictProba(self, message):
        """ returns a score dictionary that looks like: {"pos":0.8, "neg":0.2}
        """
        raise NotImplementedError("");

    pass

class DssgBinaryClassifierMajorityVote(DssgBinaryClassifier):
    """
    Majority vote classifier.
    """
    def __init__(self, posNegPrbDic):
        self._posNegPrbDic = posNegPrbDic;

    @staticmethod
    def train(binaryLabeledMessageList):
        nPos = countTruth(lambda x: x[1] == 'pos', binaryLabeledMessageList)
        nNeg = countTruth(lambda x: x[1] == 'neg', binaryLabeledMessageList)
        assert(nPos + nNeg == len(binaryLabeledMessageList));
        posNegPrbDic = {'pos': float(nPos) / (nPos+nNeg), 'neg': float(nNeg) / (nPos+nNeg)}
        
        dssgClassifier = DssgBinaryClassifierMajorityVote(posNegPrbDic);
        return dssgClassifier;
    
    def predictScore(self, message):
        """
        This turns probability into score in range [-.5, 5]
        """
        scoreDic = {};
        scoreDic['pos'] = (self._posNegPrbDic['pos']-.5)
        scoreDic['neg'] = (self._posNegPrbDic['neg']-.5)
        return self._posNegPrbDic;

    def __repr__(self):
        return self.__class__.__name__ + '(pos = %.3f, neg = %.3f)'%(self._posNegPrbDic['pos'],self._posNegPrbDic['neg'])

    pass

# class DssgBinaryClassifierNaiveBayes(DssgBinaryClassifier):
#     def __init__(self, nbClassifier):
#         self._nbClassifier = nbClassifier;
#         pass
# 
#     @classmethod
#     def train(cls, binaryLabeledMessageList):
#         tmp = map(lambda x: (cls._getFeatures(x[0]),x[1]), binaryLabeledMessageList);
#         nbClassifier = NaiveBayesClassifier.train(tmp);
#         dssgClassifier = DssgBinaryClassifierNaiveBayes(nbClassifier)
#         return dssgClassifier;
#     
#     def predictScore(self, message):
#         probs = self._nbClassifier.prob_classify(self._getFeature(message))
#         return {'pos':probs.prob('pos'), 'neg':probs.prob('neg')}
# 
#     @staticmethod
#     def _getFeatures(text):
#         return bagOfWordsExceptStopwords(getWords(text.lower()))
# 
#     pass


class DssgBinaryClassifierSVC(DssgBinaryClassifier):
    def __init__(self, classifier, dssgVectorizer, plattModel):
        assert(isinstance(dssgVectorizer, DssgVectorizer))
        self._classifier = classifier
        self._dssgVectorizer = dssgVectorizer
        self._plattModel = plattModel

    @classmethod
    def train(cls, binaryLabeledMessageList, dssgVectorizer, balance=False, C=1.0, tol=1e-3):
        msgList = map(lambda x: x[0], binaryLabeledMessageList)
        y = np.array(map(lambda x: 1 if x[1] == 'pos' else 0, binaryLabeledMessageList));
        X = dssgVectorizer.fitTransform(msgList);
        
        if (balance==True):
            classifier = LinearSVC(C=C, loss='l2', penalty='l1', dual=False, tol=tol,\
                    random_state=0, class_weight='auto')
        else:
            classifier = LinearSVC(C=C, loss='l2', penalty='l1', dual=False, tol=tol,\
                    random_state=0)

        classifier.fit(X, y)

        #- learn sigmoid
        yDeci = classifier.decision_function(X);
        yy = [1 if v == 1 else -1 for v in y]
        [A,B] = platt.SigmoidTrain(yDeci, yy)
        plattModel = [A,B];

        #- create dssgClassifier
        dssgClassifier = cls(classifier, dssgVectorizer, plattModel);
        return dssgClassifier;

    def predictScore(self, message):
        featVec = self._dssgVectorizer.transform([message]);
        df = self._classifier.decision_function(featVec)
        scoreDic = {};
        scoreDic['pos'] = df[0];
        scoreDic['neg'] = -df[0];
        return scoreDic;

    def predictProba(self, message):
        featVec = self._dssgVectorizer.transform([message]);
        df = self._classifier.decision_function(featVec)
        prb = platt.SigmoidPredict(df, self._plattModel)
        probaDic = {}
        probaDic['pos'] = prb;
        probaDic['neg'] = 1-prb;
        return probaDic;
        assert False 

 
    def __repr__(self):
        return self.__class__.__name__ + '(_vectorizer=%s, _classifier=%s)'%(self._dssgVectorizer, self._classifier)

    pass

class DssgBinaryClassifierLogisticRegression(DssgBinaryClassifier):
    def __init__(self, classifier, dssgVectorizer):
        assert(isinstance(dssgVectorizer, DssgVectorizer));
        self._classifier = classifier;
        self._dssgVectorizer = dssgVectorizer;

    @classmethod
    def train(cls, binaryLabeledMessageList, dssgVectorizer, balance=False, C=1.0, tol=1e-3):
        msgList = map(lambda x: x[0], binaryLabeledMessageList)
        y = np.array(map(lambda x: 1 if x[1] == 'pos' else 0, binaryLabeledMessageList));
        X = dssgVectorizer.fitTransform(msgList);
        
        if (balance==True):
            classifier = LogisticRegression(C=C, penalty="l2", dual=False, tol=tol, random_state=0, class_weight='auto')
        else:
            classifier = LogisticRegression(C=C, penalty="l2", dual=False, tol=tol, random_state=0)

        classifier.fit(X, y)

        dssgClassifier = cls(classifier, dssgVectorizer);
        return dssgClassifier;

    def predictScore(self, message):
        featVec = self._dssgVectorizer.transform([message]);
        df = self._classifier.decision_function(featVec)
        scoreDic = {};
        scoreDic['pos'] = df[0];
        scoreDic['neg'] = -df[0];
        return scoreDic;

    def predictProba(self, message):
        featVec = self._dssgVectorizer.transform([message]);
        ret = self._classifier.predict_proba(featVec)
        probaDic = {}
        probaDic['pos'] = ret[0][1];
        probaDic['neg'] = ret[0][0];
        return probaDic;

    def __repr__(self):
        return self.__class__.__name__ + '(_vectorizer=%s, _classifier=%s)'%(self._dssgVectorizer, self._classifier)

    pass


class DssgBinaryClassifierAdaptiveInterpolation(DssgBinaryClassifier):
    """
    predicts by "(1-alpha)*f_{global} * alpha*f_{local}"
    """
    def __init__(self, globalClassifier, localClassifier, alpha, trainLog):
        self._globalClassifier = globalClassifier;
        self._localClassifier = localClassifier;
        self._alpha = alpha;
        self._trainLog = trainLog;

    @classmethod
    def train(cls, binaryLabeledMessageList, globalClassifier, binaryClassifierTrainer):
        trainLog = ""
        nFolds = 5;

        msgList = map(lambda x: x[0], binaryLabeledMessageList)
        y = np.array(map(lambda x: 1 if x[1] == 'pos' else 0, binaryLabeledMessageList));

        #--- obtain global decision function list
        globalDFList = [];
        for msg in msgList:
            v = globalClassifier.predictScore(msg)
            globalDFList.append(v['pos'])

        tt = tic();
        #--- obtain local decision function list (by cross-prediction)
        localClassifier = None;
        if (sum(y) < 2):
            #--- if there are less than 2 data points, set alpha=0
            trainLog += "There are less than 2 data points. setting maxAlpha=0\n"
            maxAlpha=0
        else:
            #--- split into 5 and cross-predict decision function
            localDFList = [float('nan')]*len(binaryLabeledMessageList);
            folds = StratifiedKFold(y, nFolds);
            for trainIdx, testIdx in folds:
                train = [binaryLabeledMessageList[i] for i in trainIdx]
                test = [binaryLabeledMessageList[i] for i in testIdx]

                tmpClassifier = binaryClassifierTrainer(train)

                curDFList = [];
                for x in test:
                    v = tmpClassifier.predictScore(x[0])
                    curDFList.append(v['pos'])

                for i in range(len(test)):
                    localDFList[testIdx[i]] = curDFList[i]
            assert(countTruth(lambda x: np.isnan(x), localDFList) == 0)

            #--- try many alpha's, and measure F1
            f1List = []
            alphaList = np.linspace(0.0, 1.0, 101);
            for alpha in alphaList:
                combinedDFList = [ (1-alpha)*globalDFList[i] + alpha*localDFList[i] 
                        for i in range(len(globalDFList))]
                predy = [1 if v>=0.0 else 0 for v in combinedDFList];
                #- measure F1
                f1 = metrics.f1_score(y, predy);
                f1List.append(f1);

            #--- pick alpha that maximizes F1
            trainLog += "Finding alpha maximizing F1...\n"
            maxF1 = np.max(f1List);
            maxIdx = np.where(np.array(f1List) == maxF1);
            maxIdx = maxIdx[0];
            assert(len(maxIdx) != 0);
            candidateMaxAlpha = [alphaList[i] for i in maxIdx]           
            maxAlpha = np.median(candidateMaxAlpha)

            trainLog += "max(f1List) = %.3f\n"%maxF1
            trainLog += "argmax_alpha = \n%s\n"%str(np.array([alphaList[i] for i in maxIdx]))
            trainLog += "chosen alpha by median = %.3f\n"%maxAlpha;
    
            #--- train local classifier using all data 
            localClassifier = binaryClassifierTrainer(binaryLabeledMessageList)

        elapsed = toc(tt)
        trainLog += "Time taken for training: %.3f (sec)\n" % elapsed

        dssgClassifier = cls(globalClassifier, localClassifier, maxAlpha, trainLog);
        return dssgClassifier;

    def predictScore(self, message):
        if (self._localClassifier == None):
            return self._globalClassifier.predictScore(message);
        else:
            #--- compute interpolated decision value
            globalDecisionFunction = self._globalClassifier.predictScore(message)['pos'];
            localDecisionFunction = self._localClassifier.predictScore(message)['pos']; 
            df = (1-self._alpha)*globalDecisionFunction + self._alpha*localDecisionFunction

            #--- return dictionary of score
            scoreDic = {};
            scoreDic['pos'] = df;
            scoreDic['neg'] = -df;
            return scoreDic;

    #--- TODO HOW to do probablistic transformation?; I can still use Platt's method!!
    pass

class DssgBinaryClassifierAdaptiveSVC(DssgBinaryClassifier):
    """
    predicts by a training SVM on feature vector like [f_{global}, f_{local}]
    """
    def __init__(self, globalClassifier, localClassifier, metaClassifier, trainLog):
        self._globalClassifier = globalClassifier;
        self._localClassifier = localClassifier;
        self._metaClassifier = metaClassifier;
        self._trainLog = trainLog;

    @classmethod
    def train(cls, binaryLabeledMessageList, globalClassifier, binaryClassifierTrainer):
        trainLog = ""
        nFolds = 5;

        msgList = map(lambda x: x[0], binaryLabeledMessageList)
        y = np.array(map(lambda x: 1 if x[1] == 'pos' else 0, binaryLabeledMessageList));

        #--- obtain global decision function list
        globalDFList = [];
        for msg in msgList:
            v = globalClassifier.predictScore(msg)
            globalDFList.append(v['pos'])

        tt = tic();
        #--- obtain local decision function list (by cross-prediction)
        metaClassifier = None;
        if (sum(y) < 2):
            #--- if there are less than 2 data points, set alpha=0
            trainLog += "There are less than 2 data points. using globalClassifier as the classifier\n"
            localClassifier=None;
        else:
            #--- split into 5 and cross-predict decision function
            localDFList = [float('nan')]*len(binaryLabeledMessageList);
            folds = StratifiedKFold(y, nFolds);
            for trainIdx, testIdx in folds:
                train = [binaryLabeledMessageList[i] for i in trainIdx]
                test = [binaryLabeledMessageList[i] for i in testIdx]

                tmpClassifier = binaryClassifierTrainer(train)

                curDFList = [];
                for x in test:
                    v = tmpClassifier.predictScore(x[0])
                    curDFList.append(v['pos'])

                for i in range(len(test)):
                    localDFList[testIdx[i]] = curDFList[i]
            assert(countTruth(lambda x: np.isnan(x), localDFList) == 0)

            #--- train SVC!!
            #- prepare X
            X = np.hstack([np.matrix(globalDFList).T, np.matrix(localDFList).T])
            metaClassifier = LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-6,\
                    random_state=0)
            metaClassifier.fit(X,y)
            trainLog += "Trained a meta classifier:\n";
            coef = metaClassifier.coef_[0];
            trainLog += "  coef for (global, local): (%.6f, %.6f)\n"%(coef[0],coef[1])
            trainLog += "  intercept: %.6f\n" % (metaClassifier.intercept_)

            #--- train local classifier using all data 
            localClassifier = binaryClassifierTrainer(binaryLabeledMessageList)
            trainLog += "Trained a meta classifier"

        elapsed = toc(tt)
        trainLog += "Time taken for training: %.3f (sec)\n" % elapsed

        dssgClassifier = cls(globalClassifier, localClassifier, metaClassifier, trainLog);
        return dssgClassifier;

    def predictScore(self, message):
        #--- if localClassifier is None, dont use it!!
        if (self._localClassifier == None):
            return self._globalClassifier.predictScore(message); 
        else:
            #--- compute feature vector
            globalDecisionFunction = self._globalClassifier.predictScore(message)['pos'];
            localDecisionFunction = self._localClassifier.predictScore(message)['pos']; 
            testX = np.matrix([globalDecisionFunction, localDecisionFunction])
            df = self._metaClassifier.decision_function(testX);

            #--- return dictionary of score
            scoreDic = {};
            scoreDic['pos'] = df;
            scoreDic['neg'] = -df;
        return scoreDic;
    pass

################################################################################
# Category Classifiers
################################################################################

class DssgCategoryClassifier(object):
    _classifierDic = {};
    _categoryList = [];
    _trainStats = {};

    def getTrainStats(self):
        return self._trainStats;
    
    def __init__(self, classifierDic, categoryList, trainStats):
        self._classifierDic = classifierDic;
        self._categoryList = categoryList;
        self._trainStats = trainStats;

    def __repr__(self):
        return self.__class__.__name__ + '(_categoryList=%s, _classifierDic=%s)'%(self._categoryList, pprint.pformat(self._classifierDic))

    @classmethod
    def train(cls, binaryClassifierTrainer, messageList, doCV=True, verbose=True, globalBinaryClassifierDic = {}):
        """ A constructor for a classifier.
        messageList is a list of message in dictionary. Major keys are 'title', 'description', and 'categories', and so on.
        """

        logger = logging.getLogger(__name__);
        ch = logging.StreamHandler()
        logger.setLevel(logging.INFO if verbose else logging.WARNING);

        #--- Empty variables
        classifierDic = {};

        #--- extract category list in dataset
        categorySet = set();
        for msg in messageList:
            categorySet.update(msg['categories']);
        #- there might be extra categories in globalBinaryClassifierDic
        categorySet.update(globalBinaryClassifierDic.keys())
        categoryList = sorted(list(categorySet))

        #--- if we do not see certain categories in the train set
        #- we will use global classifier since we can't train
        extraCategories = set(globalBinaryClassifierDic.keys()) - set(categoryList)
        for cat in extraCategories:
            classifierDic[cat] =  globalBinaryClassifierDic[cat]

        #--- for each category, train a binary classifier
        trainStats = {}
        totNTP = 0; totNFP = 0; totNTN = 0; totNFN = 0;
        for cat in categoryList:
            logger.info('Category: %s', cat);

            if (cat in globalBinaryClassifierDic):
                globalBinaryClassifier = globalBinaryClassifierDic[cat]
            else:
                globalBinaryClassifier = None
            logger.info('  globalBinaryClassifier: %s', str(globalBinaryClassifier))

            #- turn into a binary labeled data
            dset = [];
            for msg in messageList:
                binaryLabel = 'pos' if cat in msg['categories'] else 'neg'; 
                dset.append( (msg, binaryLabel) );

            #- save stat
            nPos = countTruth(lambda x: x[1] == 'pos', dset);
            nNeg = countTruth(lambda x: x[1] == 'neg', dset);
            assert(len(dset) == nPos + nNeg);

            v = float(nPos)/(nPos+nNeg)
            mvAcc = max([v, 1-v])
            logger.info('  nPos,nNeg = (%d / %d), majority vote acc = %.3f', nPos, nNeg, mvAcc);

            statDic = {'nPos':nPos, 'nNeg':nNeg, 'majority vote acc':mvAcc}

            #--- run cross validation to estimate performance
            cvResDic = {}
            bContinue = False
            didCV = False;
            if (doCV):
                try:
                    cvResDic = dssgRunCV(binaryClassifierTrainer, 
                                         dset, 
                                         globalBinaryClassifier=globalBinaryClassifier)
                    strList = ('nFolds', 'mean of accuracy', '95% confidence interval', 'precision', 'recall', 'f1')
                    msg = '%2d-CV: %s = %.3f, %s = [%.3f, %.3f]\n' % \
                            (cvResDic[strList[0]], strList[1], cvResDic[strList[1]], strList[2], cvResDic[strList[2]][0], cvResDic[strList[2]][1])
                    for i in [1,3,4,5]:
                        msg += '       %s = %.3f' % (strList[i], cvResDic[strList[i]]);
                        if (i != 5): msg += '\n';
                    didCV = True;
                except TooSkewedLabelsException:
                    msg = 'CV accuracy estimate is not available due to skewed labels. Skipping'
#                     bContinue = True;

                logger.info('  '+msg);
#                 if (bContinue):
#                     continue;

            statDic['CV stat'] = cvResDic;
            trainStats[cat] = statDic;

            #--- build a classifier using all data.
            if (globalBinaryClassifier != None):
                dssgClassifier = binaryClassifierTrainer(dset, globalBinaryClassifier);
            else:
                if (nPos == 0 or nNeg == 0):
                    #- when only one label presents, do majority vote..
                    dssgClassifier = DssgBinaryClassifierMajorityVote.train(dset);
                else:
                    dssgClassifier = binaryClassifierTrainer(dset);

            #- report vocabulary
            if (isinstance(dssgClassifier, DssgBinaryClassifierAdaptiveInterpolation) or\
                isinstance(dssgClassifier, DssgBinaryClassifierAdaptiveSVC) or\
                isinstance(dssgClassifier, DssgBinaryClassifierMajorityVote)
                ):
                logger.info('  Can\'t get nVoca since the classifier is: %s', repr(dssgClassifier)); # TODO print out the type of classifier
            else:
                nVoca = len(dssgClassifier._dssgVectorizer.getFeatureNameList())
                logger.info('  nVoca = %d', nVoca);
                if (isinstance(dssgClassifier._dssgVectorizer, DssgMultiVectorizer)):
                    logger.info('  Vectorizer: %s', repr(dssgClassifier._dssgVectorizer))
#                     featureInfo = dssgClassifier._dssgVectorizer.getFeatureInfo()
#                     logger.info('  Multi Vectorizer: %s', str(featureInfo));

            #- add classifier to classifier dictionary.
            classifierDic[cat] = dssgClassifier;

            logger.info('  type = %s', str(type(dssgClassifier)))
            if (didCV):
                totNTP += statDic['CV stat']['nTP']
                totNFP += statDic['CV stat']['nFP']
                totNTN += statDic['CV stat']['nTN']
                totNFN += statDic['CV stat']['nFN']
        
        if (doCV):
            didCVCategoryList = []
            for cat,v in trainStats.iteritems():
                if (len(v['CV stat']) != 0):
                    didCVCategoryList.append(cat);
            logger.info('DID CV for categories: %s', str(didCVCategoryList));

            meanF1 = np.mean(np.array(map(lambda cat: trainStats[cat]['CV stat']['f1'], didCVCategoryList)))
#            meanF1 = np.mean(np.array(map(lambda x: x['CV stat']['f1'], trainStats.values())))
            logger.info('Overall mean F1 = %.3f', meanF1);

            totPrecision = float(totNTP) / (totNTP + totNFP)
            totRecall = float(totNTP) / (totNTP + totNFN)
            logger.info('Overall aggregate prec = %.3f, recall = %.3f, F1 = %.3f', totPrecision, totRecall, sp.stats.hmean([totPrecision, totRecall]));

        return cls(classifierDic, categoryList, trainStats)
        
    def predictScore(self, message):
        """
        Predicts labels in a form of {'label1':0.9, 'label2':0.8, ...}
        """

        labelProbDic = {}
        for cat in self._categoryList:
            probs = self._classifierDic[cat].predictScore(message);
            labelProbDic[cat] = probs['pos'];

        return labelProbDic;

    def predictProba(self, message):
        """
        Predicts labels in a form of {'label1':0.9, 'label2':0.8, ...}
        """

        labelProbDic = {}
        for cat in self._categoryList:
            probs = self._classifierDic[cat].predictProba(message);
            labelProbDic[cat] = probs['pos'];

        return labelProbDic;

    pass # end of class

################################################################################
# Runs CV.
################################################################################
def dssgRunCV(binaryClassifierTrainer, 
              dsetBinary, 
              nFolds=5, 
              globalBinaryClassifier=None):
    """
    raise TooSkewedLabelsException when CV can't be done

    """

    #--- Get staritified folds
    y = map(lambda x: 1 if x[1]=='pos' else 0, dsetBinary);

    #- exception: if there is num(y==1)< 2 or num(y==0)<2, can't do CV.
    ySum = sum(y);
    if ySum < 2 or len(y)-ySum < 2:
        raise TooSkewedLabelsException();
    
    folds = StratifiedKFold(y, nFolds);

    #--- Compute accuracy
    accList = []; precList = []; recallList = []; f1List = [];
    allTruey = [x[1] for x in dsetBinary];
    allPredy = [float('nan')]*len(dsetBinary);
    for trainIdx, testIdx in folds:
        train = [dsetBinary[i] for i in trainIdx]
        test = [dsetBinary[i] for i in testIdx]

        #-- train
        if (globalBinaryClassifier != None):
            binaryClassifier = binaryClassifierTrainer(train, globalBinaryClassifier);
        else:
            binaryClassifier = binaryClassifierTrainer(train)

        #-- compute accuracy (or other stats?)
        predy = [];
        for x in test:
            predScore = binaryClassifier.predictScore(x[0])
            predy.append('pos' if predScore['pos'] >= 0 else 'neg')

        truey = [x[1] for x in test];
        acc = metrics.accuracy_score(truey, predy);
        accList.append(acc);

        #-
        for i in range(len(test)):
            allPredy[testIdx[i]] = predy[i]

    #- compute prec, recall, f1
    prec = metrics.precision_score(allTruey, allPredy, pos_label='pos');
    recall = metrics.recall_score(allTruey, allPredy, pos_label='pos');
    f1 = metrics.f1_score(allTruey, allPredy, pos_label='pos');
    nTP = sum([allTruey[i] == 'pos' and allPredy[i] == 'pos' for i in range(len(allTruey))]);
    nFP = sum([allTruey[i] == 'neg' and allPredy[i] == 'pos' for i in range(len(allTruey))]);
    nTN = sum([allTruey[i] == 'neg' and allPredy[i] == 'neg' for i in range(len(allTruey))]);
    nFN = sum([allTruey[i] == 'pos' and allPredy[i] == 'neg' for i in range(len(allTruey))]);

    resDic = {};
    resDic['nFolds'] = nFolds;
    me = np.mean(accList);
    resDic['mean of accuracy'] = me;
    stderr = stats.sem(accList);
    resDic['standard error of accuracy'] = stderr;
    alpha = 0.05; # failure rate
    dev = stderr*stats.t.ppf(1-alpha/2, len(accList) - 1);
    resDic['95% confidence interval'] = (me - dev, me + dev)
    resDic['precision'] = prec
    resDic['recall'] = recall
    resDic['f1'] = f1
    resDic['nTP'] = nTP
    resDic['nFP'] = nFP
    resDic['nTN'] = nTN
    resDic['nFN'] = nFN

    return resDic;

