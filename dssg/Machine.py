# -*- coding: utf-8 -*-
from .dssgclassifier import *
# do easy_install python-hashes to install (or visit
# https://github.com/sangelone/python-hashes)
from hashes.simhash import simhash
from pprint import pprint

# import MicrosoftTranslatorApi
import guess_language
import langid

import nltk
import re

import cPickle as pickle

LANGUAGE_GUESS_OPTIONS = [
    'microsoft',
    'python_guess_language',
    'langid',
    'all'
]
LANGUAGE_GUESS_METHOD = LANGUAGE_GUESS_OPTIONS[2]

LOCATION_ENTITIES = ['LOCATION', 'GPE', 'GSP']


class Machine(object):

    """Base class for Machine Learning"""

    _categories = []
    _entities = []
    _categoryClassifier = None
    _status = []
    _likely_languages = []

    def __init__(self):
        pass

    #
    # Model Setup: training, import, export
    #

    @classmethod
    def load(cls, infile=""):
        """
        Allows the user to import an existing model (e.g. election, 
        natural disaster, etc).

        These models will have a set of starting categories, e.g.
            - model: election
                - category: polling administration issues
                - category: ...
            - model: natural disaster
                - category: ...
                - category: ...

        input: path to the model file [string]
        output: none
        errors: AssertionError if unpickled object is not valid class
        """
        f = open(infile, 'r')
        mac = pickle.load(f)
        assert(isinstance(
            mac, cls)), "Type of unpickled object must be Machine"
        return mac

    def save(self, outfile=""):
        """
        Allows the user to export the current model (useful after it has been 
        trained on new labeled message data)

        input: path to the model file
        output: none
        """
        f = open(outfile, 'w')
        pickle.dump(self, f)
        f.close()
        return

    def train(self, messageList):
        """Takes list of messages. each message is a dictionary.
        """

        #-TODO change cleanup codes so it uses dictionary form of message
# Cleanup training dataset
#         params = {
#             'remove_junk_messages' : False,
#             'use_parent_categories_only' : False,
#             'remove_duplicate_categories' : False,
#         }
#         labeledMessages = self.clean_data(labeledMessages, params)

        #--- save to _messages, and compute their hashcodes for simhash
        #- TODO concatenate title??
        self._messageMap = dict(((v['id'], v) for v in messageList))
        self._simhashList = [(v['id'], simhash(unicodeToAscii(
            v['description']))) for v in messageList]

#        self._messages = map(lambda x: x['description'], messageList);
# self._simhashes = map(lambda x:
# simhash(unicodeToAscii(x['description'])), messageList);

        #--- collect category list
        categorySet = set()
        for msg in messageList:
            categorySet.update(msg['categories'])

        #--- update categories
        categories = sorted(list(categorySet))

        #--- train classifiers
        minFreq = 5
        # 1
        unigramExtractor = DssgUnigramExtractor()

        def dssgVectorizerGenerator():
            return DssgVectorizerUnigramCount(unigramExtractor, minFreq)

        def dssgBinaryClassifierTrainer(train):
            return DssgBinaryClassifierSVC.train(
                train, dssgVectorizerGenerator(), balance=False)

        categoryClassifier = DssgCategoryClassifier.train(
            dssgBinaryClassifierTrainer, messageList, dssgVectorizerGenerator)
        self._categoryClassifier = categoryClassifier
        return self

    #
    # Methods
    #
    def getTrainStats(self):
        return self._categoryClassifier.getTrainStats()

    def guess(self, messages):
        """ Takes a list of messages (each in dictionary form), and make
        guesses their category labels, languages, and so on.
        :param messages list of messages
        :return [(msgDict, {'c1': 4.6, 'c2': 4.2}), ...], list of pairs:
            message, dictionary of categories.
            The numbers returned range from (-inf, inf), but are ranks
            and cannot be interpreted beyond (e.g., as probabilites).
        """
        assert (self._categoryClassifier is not None)
                # TODO turn it into an exception

        messages = self._listify(messages)

        similarity_threshold = .875
        # minimum_similar_messages = 5
        output = []
        for msg in messages:
            categories = self._categoryClassifier.predictScore(msg)
            similarity_computations = self.computeSimilarities(
                msg)  # TODO change..
            # pprint(similarity_computations[:10])
            similar_messages = [self._messageMap[sc[
                0]] for sc in similarity_computations if sc[1] > similarity_threshold]

            msg_metrics = {
                # 'languages': self.guess_language(),
                'categories': categories,
                # 'entities': self.guess_entities(),
                'similar_messages': similar_messages,
            }

            output.append(msg_metrics)
            # pprint(msg_metrics)

        return output

    def computeSimilarities(self, msg):
        """
        returns a set of message id's with similarity score, sorted by
        similarity score.
        
        I recommend using >=0.875 to define 'near-dup'.
        :return [('1', 0.9), ('2', 0.8), ...], sorted by the real value
                (second element of each item) in decreasing order.
        """
        simhashCode = simhash(unicodeToAscii(msg['description']))

        retList = []
        for i in range(len(self._simhashList)):
            id = self._simhashList[i][0]
            val = self._simhashList[i][1].similarity(simhashCode)
            retList.append((id, val))

        retList.sort(key=lambda x: x[1], reverse=True)
        return retList

    @staticmethod
    def guess_language(text):
        """Returns list of language guesses for the message, 
        with confidence measure (0 to 1).
        """

        return langid.rank(text)

    @staticmethod
    def guess_entities(text):
        """Returns list of non-location entity guesses for the message
        Each entity (ideally) also includes
            - text
            - start (offset in included string)
            - type (person, location, etc)
            - confidence
        """

        entities = Machine._extract_entities(text)
        for e in LOCATION_ENTITIES:
            if e in entities:
                del entities[e]

        return entities

    @staticmethod
    def guess_locations(text):
        """Returns the list of location entities contained in the specified
        text
        
        :param text: the message to be analyzed
        """
        entities = Machine._extract_entities(text)
        location_entity_set = set(LOCATION_ENTITIES)
        entity_keys = entities.keys()
        for k in entity_keys:
            if not k in location_entity_set:
                del entities[k]

        return entities
        
    #
    @staticmethod
    def guess_private_info(text, **kwargs):
        """ Returns list of potentially private/sensitive information to 
            consider stripping.
            Output is a list of tuples each containing two parts:
                * the private information type (PERSON, ID, PHONE, etc.)
                * the word(s) in the message that correspond to this unit

            Note that the same words in text may appear (in whole or part) in
            multiple tuples. For example, a number may meet the criteria for
            both an ID and a phone number.

            The types of information:

            1. Named entities [types: PERSON, GPE, etc.]
                Note this includes possible locations (GPE), which may be 
                non-private and useful for geolocation
            2. ID numbers (passport, driver's license, etc.) [type: ID]
            3. Usernames (e.g. Twitter handles) [type: USERNAME]
            4. URLs [type: URL]
            5. E-mail addresses [type: EMAIL]
            6. Phone numbers, using the optional provided regex "phone_regex" [type: PHONE]
        """

        # If phone_regex not provided, use a default
        phone_regex = None
        if "phone_regex" in kwargs.keys():
            phone_regex = kwargs["phone_regex"]

        return Machine._extract_entities(text) + \
            Machine._extract_ids(text) + \
            Machine._extract_usernames(text) + \
            Machine._extract_emails(text) + \
            Machine._extract_urls(text) + \
            Machine._extract_phones(text, phone_regex)

    @staticmethod
    def _extract_entities(text):
    # Returns named entities [tags: PERSON, GPE, etc.]
        entities_list = []
        for sent in nltk.sent_tokenize(text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'node'):
                    item = chunk.node, ' '.join(c[0] for c in chunk.leaves())
                    entities_list.append(item)

        entities = {}
        for (group, entity) in entities_list:
            if not group in entities:
                entities[group] = set([])
            entities[group].add(entity)

        return entities

    @staticmethod
    def _extract_ids(text):
    # Returns tokens that have at least one digit, and is at least four
    # characters long [tag: ID]
        ids_list = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) >= 4 and any(char.isdigit() for char in word):
                    ids_list.append(("ID", word))
        return ids_list

    @staticmethod
    def _extract_usernames(text):
    # Returns Twitter usernames of form @handle 
    # (alphanumerical and "_", max length: 15) [tag: USERNAME]
        # twitter_regex = r'\[A-Za-z0-9_]{1,15}'
        twitter_regex = r'^|[^@\w](@\w{1,15})\b'
        twitter_re = re.compile(twitter_regex)
        twitter_list = [("TWITTER", twitter) for twitter in twitter_re.findall(
            text) if twitter != ""]
        return twitter_list

    # ----------------
    @staticmethod
    def _extract_urls(text):
    # Returns URLs [tag: URL]
        url_regex = r'''
                        (?xi)
                            \b
                            (                           # Capture 1: entire matched URL
                            (?:
                                [a-z][\w-]+:                # URL protocol and colon
                                (?:
                                /{1,3}                        # 1-3 slashes
                                |                             #   or
                                [a-z0-9%]                     # Single letter or digit or '%'
                                                                # (Trying not to match e.g. "URI::Escape")
                                )
                                |                           #   or
                                www\d{0,3}[.]               # "www.", "www1.", "www2." … "www999."
                                |                           #   or
                                [a-z0-9.\-]+[.][a-z]{2,4}/  # looks like domain name followed by a slash
                            )
                            (?:                           # One or more:
                                [^\s()<>]+                      # Run of non-space, non-()<>
                                |                               #   or
                                \(([^\s()<>]+|(\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
                            )+
                            (?:                           # End with:
                                \(([^\s()<>]+|(\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
                                |                                   #   or
                                [^\s`!()\[\]{};:'".,<>?«»“”‘’]        # not a space or one of these punct chars
                            )
                        )'''

        url_re = re.compile(url_regex, re.VERBOSE)
        url_list = [("URL", url[0]) for url in url_re.findall(text)]
        return url_list

    @staticmethod
    def _extract_emails(text):
        # Returns e-mail addresses [tag: EMAIL]
        emails_regex = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}"
        emails_re = re.compile(emails_regex)
        emails_list = [("EMAIL", email) for email in emails_re.findall(text)]
        return emails_list

    @staticmethod
    def _extract_phones(text, phone_regex=None):
        # Returns phone numbers, using optional regex [tag: PHONE]

        if not phone_regex:
            # default phone regex (United States)
            phone_regex = r'''(\b
                                \d{3}     # area code is 3 digits (e.g. '800')
                                \D*         # optional separator is any number of non-digits
                                \d{3}     # trunk is 3 digits (e.g. '555')
                                \D*         # optional separator
                                \d{4}     # rest of number is 4 digits (e.g. '1212')
                                )'''

        phone_re = re.compile(phone_regex, re.VERBOSE)
        phone_list = [("PHONE", phone) for phone in phone_re.findall(text)]
        return phone_list

#    def guess_then_train(self, messages=[]):
#        """This is standard operating procedure once the classifier has been running for a bit.
#        We'll want to make our best a guess and then do incremental training.
#        :param messages list of messages
#        """
#        output = self.guess(messages)
#        self.train(messages)
#
#        return output
    def _listify(self, messages):
        """Allow argument to be a string or a list
        """

        if not isinstance(messages, (list, tuple)):
            # Make a list with one item
            messages = [messages]

        return messages

    def clean_data(self, lm, params={}):
        print "beginning to filter..."

        self._training_dataset_stats(lm)

        print "FILTER: remove 'bad' messages"
        if params['remove_junk_messages']:
            lm = self._filter_labeled_messages(lm)
            self._training_dataset_stats(lm)
        else:
            print "skipped"
        # removeUselessCategories(lm)

        print "FILTER: use parent categories only"
        if params['use_parent_categories_only']:
            lm = self._use_parent_categories_only(lm, None)
            self._training_dataset_stats(lm)
        else:
            print "skipped"

        print "FILTER: remove duplicate categories"
        if params['remove_duplicate_categories']:
            lm = self._remove_duplicate_categories(lm)
            self._training_dataset_stats(lm)
        else:
            print "skipped"

        print "...end of filtering"

        return lm

    def _filter_labeled_messages(self, lm):
        """Takes a dictionary and returns a filtered version, using rules 
        specified in filter_fn()
        """

        def _filter_fn(item):
            """Removes low value messages, so they aren't used for training.

            Possible filter rules
            - (near)-identical to a previous message (simhash)
            - message too short (len)
            - message has too few words
            - message has mixture of languages; not enough English to be 
                useful in natural language processing
            """
            message_text = item[0]
            # message_categories = item[1]

            MIN_LENGTH = 3
            if len(message_text) < MIN_LENGTH:
                # Remove these
                return False

            else:
                return True

        # return dict((k, v) for (k, v) in data_dict.iteritems() if
        # _filter_fn(v))
        return filter(_filter_fn, lm)

    def _use_parent_categories_only(self, lm, category_mapping=None):
        """
        """

        # TODO: This is hardcoded to work only with uchaguzi parent categories
        #   We also have hierarchy from 'data/processed/nyc_snow_categories.json',
        #   because drawn from MySQL db which notes parent_id
        j = json.load(open('data/processed/uchaguzi_new_categories.json', 'r'))
        mapping = {}
        for i in j:
            # default: map to self
            mapping[i['category_title']] = i['category_title']

            if i['parent_id']:
                # Get parents category name
                for t in j:
                    if t['id'] == i['parent_id']:
                        mapping[i['category_title']] = t['category_title']

                # Fails quietly... if cannot find parent, keeps self<->self
                # mapping instead of self<->parent

        # pprint(mapping)
        # print len(mapping.values())
        # print len(list(set(mapping.values())))

        def get_parent_category(c):
            return mapping[c]

        out = []
        for i in lm:
            new_categories = []
            old_categories = i[1]
            for c in old_categories:
                pc = get_parent_category(c)
                new_categories.append(pc)

            i[1] = new_categories
            out.append(i)

        return out

    def _remove_duplicate_categories(self, lm):
        """
        """
        # (k = category text, v = parent category text)

        out = []
        count = 0

        for i in lm:
            before = i[1]
            after = list(set(i[1]))
            i[1] = after

            if len(before) != len(after):
                count += 1

            out.append(i)

        print "Number of messages that had duplicate categories %s" % (count,)
        return out

    def _training_dataset_stats(self, lm):
        categories = []
        no_dup_categories = []
        for m in lm:
            categories.extend(m[1])
            no_dup_categories.extend(list(set(m[1])))

        unique_categories = list(set(categories))

        print "================================"
        print "# messages: %s" % len(lm)
        print "# category tags on all messages (including duplicates): %s" % len(categories)
        print "# category tags on all messages (excluding duplicates): %s" % len(no_dup_categories)
        print "# unique categories: %s" % len(unique_categories)
        print "================================"
