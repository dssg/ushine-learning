from .util import * 
import nltk;
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

from scipy.sparse import coo_matrix, hstack
import numpy as np

import ipdb

def getWords(text):
    return word_tokenize(text)

def bagOfWords(words):
    return dict([(word.lower(), True) for word in words])

def bagOfWordsNotInSet(words, badwords):
    return bagOfWords(set(words) - set(badwords))

def bagOfWordsExceptStopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bagOfWordsNotInSet(words, badwords)

################################################################################
# UnigramExtractor
################################################################################

class DssgUnigramExtractor(object):
    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()
        self._stopwordSet = set(stopwords.words('english'))
        self._stemmer = PorterStemmer();

    def __repr__(self):
        return self.__class__.__name__ + '()'; 

    def extract(self, text):
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace('&quot;', '"').replace('&amp;', '&').replace('&nbsp;', ' ')
        text = nltk.clean_html(text);
        tokens = self._tokenizer.tokenize(text)
        
        newTokens = []
        for tok in tokens:
            #- lowercase, remove '
            tok = tok.lower().strip("`'.,-_*/:;\\!@#$%^&*()=\""); 

            #- remove stopwords, one character word, only numbers
            #- remove one character word
            #- remove only numbers
            if (tok in self._stopwordSet or len(tok) <= 1 or isAllNumbers(tok)):
                continue

            #- apply stemming
#        oldTok = copy.deepcopy(tok); # for debug
            tok = self._stemmer.stem(tok)
            if (tok in self._stopwordSet): # sometimes a token is like 'theres' and becomes stopword after stemming
                continue;

            newTokens.append(tok);
        return newTokens;

class DssgBigramExtractor(object):
    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()
        self._stopwordSet = set(stopwords.words('english'))
        self._stemmer = PorterStemmer();

    def __repr__(self):
        return self.__class__.__name__ + '()';
    
    def extract(self, text):
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace('&quot;', '"').replace('&amp;', '&').replace('&nbsp;', ' ')
        text = nltk.clean_html(text);

        sentenceList = nltk.tokenize.sent_tokenize(text)
        bigrams = []
        for sentence in sentenceList:
            tokens = self._tokenizer.tokenize(sentence)
            
            # Preprocessing
            # print "text: %s" % (text,)
            newTokens = []
            for tok in tokens:
                #- lowercase, remove '
                tok = tok.lower().strip("`'.,-_*/:;\\!@#$%^&*()=\""); 

                #- remove stopwords, one character word, only numbers
                #- remove one character word
                #- remove only numbers
                if (tok in self._stopwordSet or len(tok) <= 1 or isAllNumbers(tok)):
                    continue

                #- apply stemming
                tok = self._stemmer.stem(tok)
                if (tok in self._stopwordSet): # sometimes a token is like 'theres' and becomes stopword after stemming
                    continue;

                newTokens.append(tok);
            
            # Bigrams
            curBigrams = nltk.bigrams(newTokens);
            bigrams += curBigrams;
        
        # print "bigrams:"
        # print bigrams
        # raise
        return bigrams

class DssgPosExtractor(object):
    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()
        self._stopwordSet = set(stopwords.words('english'))
        self._stemmer = PorterStemmer();

    def __repr__(self):
        return self.__class__.__name__ + '()';
    
    def extract(self, text):
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace('&quot;', '"').replace('&amp;', '&').replace('&nbsp;', ' ')
        text = nltk.clean_html(text);
        tokens = self._tokenizer.tokenize(text)

        assert False, "modify so it first do sent_tokenize()"
        # Preprocessing
        # print "text: %s" % (text,)
        newTokens = []
        for tok in tokens:
            #- lowercase, remove '
            # tok = tok.lower().strip("`'.,-_*/:;\\!@#$%^&*()=\""); 

            #- remove stopwords, one character word, only numbers
            #- remove one character word
            #- remove only numbers
            if (tok in self._stopwordSet or len(tok) <= 1 or isAllNumbers(tok)):
                continue

            #- apply stemming
            tok = self._stemmer.stem(tok)
            if (tok in self._stopwordSet): # sometimes a token is like 'theres' and becomes stopword after stemming
                continue;

            newTokens.append(tok);

        #--- POS Tags
        posTags = nltk.pos_tag(newTokens)
        # print "pos tags:"
        # print posTags
        # raise
        return posTags

################################################################################
# Vectorizer
################################################################################

class DssgVectorizer(object):
    def __init__(self):
        pass

    def fitTransform(self, messageList):
        #- find unigram, compute length, POS tagging and so on...
        raise NotImplementedError("");

    def transform(self, messageList):
        raise NotImplementedError("");

    def getFeatureNameList(self):
        raise NotImplementedError("");

    pass

class DssgMultiVectorizer(DssgVectorizer):
    _vectorizers = None
    _unigramCountDic = {}   # TODO: Remove this or make an accessor for this if needed, just to prevent crash in test_featureEngineering.py when calling
                            # vocaDic = vectorizer._unigramCountDic;

    def __init__(self, vectorizers=[]):
        # for v in vectorizers:
        #     print v

        self._vectorizers = vectorizers

    def __repr__(self):
        return self.__class__.__name__ + '(_vectorizers=%s)' % (repr(self._vectorizers))

    def fitTransform(self, messageList):
        return self._hstack_vectors("fitTransform", messageList)

    def transform(self, messageList):
        return self._hstack_vectors("transform", messageList)

    def _hstack_vectors(self, method, messageList):
        # create base vector, to append other vectorizations
        # Todo: pass actual method to call in vectorizer
        
        # baseVector = coo_matrix([[0]*len(messageList)]).T
        # combined_vectors = baseVector
        combined_vectors = None
        for v in self._vectorizers:
            # Todo: pass actual method to call in vectorizer
            if method == "fitTransform":
                output_vector = v.fitTransform(messageList)
            else: 
                output_vector = v.transform(messageList)
            
            if combined_vectors is None:
                combined_vectors = output_vector
            else:
                # print output_vector.todense()
                # print combined_vectors.todense()
                # ipdb.set_trace()
                combined_vectors = hstack([combined_vectors, output_vector])

        return combined_vectors

    def getFeatureNameList(self):
        # merge the who feature names lists
        # the indexes of 2nd vectorization need to be += the column count of the 1st vectorization, and so on...
        featureNameList = []
        for v in self._vectorizers:
            featureNameList.extend(v.getFeatureNameList())

        return featureNameList

    def getFeatureInfo(self):
        aList = []
        for v in self._vectorizers:
            aList.append((str(v), len(v.getFeatureNameList())))
        return aList

    pass

class DssgVectorizerTfIdf(DssgVectorizer):
    def __init__(self, ngramExtractor, minFreq=1):
        self._ngramExtractor = ngramExtractor;
        self._minFreq = minFreq;
        self._unigramCountDic = {};
        self._unigramIdxDic = {};
        self._idfDic = {};

    def __repr__(self):
        return self.__class__.__name__ + '(_ngramExtractor=%s, _minFreq=%s, len(_unigramCountDic)=%d)' % \
                (repr(self._ngramExtractor), str(self._minFreq), len(self._unigramCountDic))

    def fitTransform(self, messageList):
        assert(len(self._unigramIdxDic) == 0);

        #- find unigrams
        unigramCountDic = {};
        cache = []
        for msg in messageList: 
            unigramList = self._ngramExtractor.extract(msg['title'] + ' ' + msg['description']);
#            unigramList = self._extractUnigramFunc( msg['description']);
#            unigramList = self._extractUnigramFunc( msg['title']);
            cache.append(unigramList);
            for unigram in unigramList:
                if (unigram not in unigramCountDic):
                    unigramCountDic[unigram] = 0;
                unigramCountDic[unigram] += 1;

        #- cutoff
        pairs = unigramCountDic.items()
        pairs = filter(lambda x: x[1] >= self._minFreq, pairs);

        unigramCountDic = dict(pairs);
        unigramList = sorted(unigramCountDic.keys());
        self._unigramIdxDic =  dict(zip(unigramList, range(len(unigramList))))
        self._unigramCountDic = unigramCountDic;

        #- compute IDF
        docFreqDic = dict(zip(unigramList, [0]*len(unigramList)));
        for unigramList in cache:
            for unigram in set(unigramList): # only counts 'existence'
                if (unigram in unigramCountDic):
                    docFreqDic[unigram] += 1

        idfDic = {}
        logNDoc = np.log(len(messageList))
        for unigramList in cache:
            for unigram in unigramList:
                if (unigram in unigramCountDic):
                    idfDic[unigram] = logNDoc - np.log(docFreqDic[unigram]);
        self._idfDic = idfDic;

        #- transform
        return self.transform(messageList,cache=cache);

    def transform(self, messageList, cache=None):
        nUnigram = len(self._unigramIdxDic)
        rowNum = 0; iList = []; jList = []; valueList = []

        for i in range(len(messageList)):
            msg = messageList[i]['title'] + ' ' + messageList[i]['description'];
#            msg = messageList[i]['description'];
#            msg = messageList[i]['title'];
            row = {};
            if (cache != None): unigramList = cache[i];
            else:               unigramList = self._ngramExtractor.extract(msg);

            for unigram in unigramList:
                if (unigram in self._unigramIdxDic):
                    if (unigram not in row): row[unigram] = 1;
                    else:                    row[unigram] += 1

            #- term frequency normalization.
            if (len(row) != 0):
                maxFreq = max(row.values())
                for k in row: 
                    row[k] = 0 if maxFreq==0 else float(row[k]) / maxFreq

            for (k,v) in row.iteritems():
                iList.append(rowNum);
                jList.append(self._unigramIdxDic[k]);
                tfIdf = float(v) * self._idfDic[k];
                valueList.append(tfIdf);

            rowNum += 1;

        #- this has happened when drawing learning curve due to too small dataset. 
        nn = len(messageList);
        if (len(self._unigramIdxDic) == 0):
            retMat = coo_matrix(([0]*nn, (range(nn), [0]*nn)), shape=(nn, 1));
        else:
            retMat = coo_matrix((valueList, (iList, jList)), \
                    shape=(len(messageList), len(self._unigramIdxDic)));
        return retMat;

    def getFeatureNameList(self):
        return self._unigramIdxDic.keys()

    pass

 
class DssgVectorizerCount(DssgVectorizer):
    def __init__(self, ngramExtractor, minFreq=1):
        self._ngramExtractor = ngramExtractor;
        self._minFreq = minFreq;
        self._ngramCountDic = {};
        self._ngramIdxDic = {};

    def __repr__(self):
        return self.__class__.__name__ + '(_ngramExtractor=%s, _minFreq=%s, len(_ngramCountDic)=%d)' % \
                (repr(self._ngramExtractor), str(self._minFreq), len(self._ngramCountDic))

    def fitTransform(self, messageList):
        assert(len(self._ngramIdxDic) == 0);

        #- find unigrams
        unigramCountDic = {};
        cache = []
        for msg in messageList: 
            unigramList = self._ngramExtractor.extract(msg['title'] + ' ' + msg['description']);
#            unigramList = self._extractUnigramFunc( msg['description']);
#            unigramList = self._extractUnigramFunc( msg['title']);
            cache.append(unigramList);
            for unigram in unigramList:
                if (unigram not in unigramCountDic):
                    unigramCountDic[unigram] = 0;
                unigramCountDic[unigram] += 1;

        #- cutoff
        pairs = unigramCountDic.items()
        pairs = filter(lambda x: x[1] >= self._minFreq, pairs);

        unigramCountDic = dict(pairs);
        unigramList = sorted(unigramCountDic.keys());
        self._ngramIdxDic =  dict(zip(unigramList, range(len(unigramList))))
        self._ngramCountDic = unigramCountDic;

        #- transform
        return self.transform(messageList,cache=cache);

    def transform(self, messageList, cache=None):
        nUnigram = len(self._ngramIdxDic)
        rowNum = 0; iList = []; jList = []; valueList = []

        for i in range(len(messageList)):
            msg = messageList[i]['title'] + ' ' + messageList[i]['description'];
#            msg = messageList[i]['description'];
#            msg = messageList[i]['title'];
            row = {};
            if (cache != None): unigramList = cache[i];
            else:               unigramList = self._ngramExtractor.extract(msg);

            for unigram in unigramList:
                if (unigram in self._ngramIdxDic):
                    if (unigram not in row): row[unigram] = 1;
                    else:                    row[unigram] += 1
            for (k,v) in row.iteritems():
                iList.append(rowNum);
                jList.append(self._ngramIdxDic[k]);
                valueList.append(v);

            rowNum += 1;

        #- this has happened when drawing learning curve due to too small dataset. 
        nn = len(messageList);
        if (len(self._ngramIdxDic) == 0):
            retMat = coo_matrix(([0]*nn, (range(nn), [0]*nn)), shape=(nn, 1));
        else:
            retMat = coo_matrix((valueList, (iList, jList)), \
                    shape=(len(messageList), len(self._ngramIdxDic)));
        return retMat;
    
    def getFeatureNameList(self):
        return self._ngramIdxDic.keys()

    pass


class DssgVectorizerTopics(DssgVectorizer):
    def __init__(self, ngramExtractor, minFreq=1):
        self._ngramExtractor = ngramExtractor;
        self._minFreq = minFreq;
        self._unigramCountDic = {};
        self._unigramIdxDic = {};
        self._topicModel = None

    def __repr__(self):
        return self.__class__.__name__ + '(_ngramExtractor=%s, _minFreq=%s, len(_unigramCountDic)=%d)' % \
                (repr(self._ngramExtractor), str(self._minFreq), len(self._unigramCountDic))

    def fitTransform(self, messageList):
        assert(len(self._unigramIdxDic) == 0);

        #- find unigrams
        unigramCountDic = {};
        # cache = []
        cache = range(topic_count)

        # Generate Topic Model from Message List
        # CREATE TEMP CSV
            # msg['id'], msg['title'] + ' ' + msg['description']
            # self._topicModel = pymallet('path/to/csv')
            # topic_matrix = self._topicModel.generate_topic_weights(topic_count=5)
        topic_matrix = [[1,.1,.9], [2,.3,.7]]
        print topic_matrix
        return


        #
        for msg in messageList: 
            topic_weights = topic_matrix[message_id][1:] # skip the item, include its topics

            for i in range(len(topic_weights)):
                unigramCountDic[i] = topic_weights[i]

            unigramList = self._ngramExtractor.extract();
            cache.append(unigramCountDic.keys()); #topic ids are 0-to-n, where n=topic_count


        #- cutoff
        pairs = unigramCountDic.items()
        # pairs = filter(lambda x: x[1] >= self._minFreq, pairs);

        unigramCountDic = dict(pairs);
        unigramList = sorted(unigramCountDic.keys());
        self._unigramIdxDic =  dict(zip(unigramList, range(len(unigramList))))
        self._unigramCountDic = unigramCountDic;

        #- transform
        return self.transform(messageList,cache=cache);

    def transform(self, messageList, cache=None):
        # TODO: Need to update Pymallet to allow making a "guess" using existing topics on new input
        pass
        
        nUnigram = len(self._unigramIdxDic)
        rowNum = 0; iList = []; jList = []; valueList = []

        for i in range(len(messageList)):
            msg = messageList[i]['title'] + ' ' + messageList[i]['description'];
#            msg = messageList[i]['description'];
#            msg = messageList[i]['title'];
            row = {};
            if (cache != None): unigramList = cache[i];
            else:               unigramList = self._ngramExtractor.extract(msg);

            for unigram in unigramList:
                if (unigram in self._unigramIdxDic):
                    if (unigram not in row): row[unigram] = 1;
                    else:                    row[unigram] += 1
            for (k,v) in row.iteritems():
                iList.append(rowNum);
                jList.append(self._unigramIdxDic[k]);
                valueList.append(v);

            rowNum += 1;

        #- this has happened when drawing learning curve due to too small dataset. 
        nn = len(messageList);
        if (len(self._unigramIdxDic) == 0):
            retMat = coo_matrix(([0]*nn, (range(nn), [0]*nn)), shape=(nn, 1));
        else:
            retMat = coo_matrix((valueList, (iList, jList)), \
                    shape=(len(messageList), len(self._unigramIdxDic)));
        return retMat;
    
    def getFeatureNameList(self):
        return self._unigramIdxDic.keys()

    pass


class DssgVectorizerUnigramBySklearn(DssgVectorizer):
    def __init__(self, minFreq=1):
        self._vectorizer = None;
        pass

    def __repr__(self):
        return self.__class__.__name__ + '(_vectorizer=%s)' % \
                (repr(self._vectorizer))

    def fitTransform(self, messageList):
        assert(self._vectorizer == None);
        self._vectorizer = CountVectorizer(min_df=1,stop_words='english', token_pattern=u'(?u)\\b\\w*[a-zA-Z]\\w*\\b')
        return self._vectorizer.fit_transform([msg['title'] + ' ' + msg['description'] for msg in messageList]);

    def transform(self, messageList):
        return self._vectorizer.transform([msg['title'] + ' ' + msg['description'] for msg in messageList]);

    def getFeatureNameList(self):
        return self._vectorizer.vocabulary_.keys()

    pass

