import csv
import json
import codecs
import copy
import logging
import os

uchaguziPath = os.path.abspath(
    '.') + '/data/processed/uchaguzi/uchaguzi-message_with_category_list.csv'
uchaguziJsonPath = os.path.abspath('.') + '/data/processed/uchaguzi_new.json'
uchaguziCategoryJsonPath = os.path.abspath(
    '.') + '/data/processed/uchaguzi_new_categories.json'


def UnicodeDictReader(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield dict([(key, unicode(value, 'utf-8')) for key, value in row.iteritems()])


def unicodeToAscii(unicodeStr):
    s = copy.copy(unicodeStr)
    while True:
        try:
            ret = codecs.encode(s, 'ascii')
            return ret
        except UnicodeEncodeError as e:
            s = s.replace(e.object[e.start:e.end], ' ')
    return None


def loadIncidentListFromCsv(path):
    """
    returns a list of incidents. An incident is [incidentId, message, categoryIdList, categoryTitleList].
    However, the categories may have duplicates. Duplicates are not meaningful and should be
    removed, but I simply did not remove them in this function.
    """
    incidentList = []
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        bFirstLine = True
        for row in reader:
            if (bFirstLine):  # skip the first line
                bFirstLine = False
                continue
            assert(len(row) == 4)
            incidentId = int(row[0])
            message = row[1]
            # Note that there are duplicates.
            categoryIdList = map(lambda x: int(x), row[2].split(','))
            categoryTitleList = map(lambda x: x.strip(), row[3].split(','))
            assert(len(categoryIdList) == len(categoryTitleList))

            incidentList.append(
                [incidentId, message, categoryIdList, categoryTitleList])
    return incidentList


def getFullMessagesFromJson(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
        messageList = []
        keyList = sorted(data.keys())  # - sort by keys.
        for k in keyList:
            v = data[k]
            v[u'id'] = k
            messageList.append(v)
        return messageList
#
# def getLabeledMessagesFromJson(path):
#    with open(path,'r') as fp:
#        j = json.load(fp)
#        allItems = []
#        for key in j:
#            item = j[key]
#            allItems.append( [ item['description'], item['categories'] ] )
#        return allItems


def countTruth(boolFunc, aList):
    return len(filter(boolFunc, aList))


def isAllNumbers(aStr):
    aStr = aStr.strip()
    if (len(aStr) == countTruth(lambda x: x >= '0' and x <= '9', aStr)):
        return True
    else:
        return False


def loadJsonFromPath(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


# def getLabeledMessagesFromUchaguziMergedCategory(uchaguziJsonPath, uchaguziCategoryJsonPath):
#    labeledMessageList = getLabeledMessagesFromJson(uchaguziJsonPath)
# - map 'Polling station logisitcal issues' to 'Polling Station Logisitcal Issues'
# - remove duplicate category labels
#    for msg in labeledMessageList:
#        for j in range(len(msg[1])):
#            if msg[1][j] == 'Polling station logisitcal issues':
#                msg[1][j] = 'Polling Station Logistical Issues'
#        msg[1] = list(set(msg[1]));
#
# --- these are selected categories. let's transform
#    selectedCategories = [('parent', 'Counting + Results'),\
#                          ('parent', 'Fear and Tension'),\
#                          ('parent', 'POSITIVE EVENTS'),\
#                          ('parent', 'Polling Station Administration'),\
#                          ('parent', 'Security Issues'),\
#                          ('parent', 'Staffing Issues'),\
#                          ('parent', 'Voting Issues'),\
#                          ('leaf',   'Resolved'),\
#                          ('leaf',   'Unresolved')]
#
#    categories = loadJsonFromPath(uchaguziCategoryJsonPath);
#    categoryByName = dict( [(cat['category_title'], cat) for cat in categories])
#    categoryById = dict( [(cat['id'], cat) for cat in categories])
#
# --- create mappings
#    catMap = {}
#    for selectedCat in selectedCategories:
#        catType = selectedCat[0]; catName = selectedCat[1];
#        id = categoryByName[catName]['id'];
#        if (catType == 'parent'):
# - find all categories that falls below it, or itself.
#            for item in categories:
#                if (item['parent_id'] == id or item['id'] == id):
#                    catMap[item['category_title']] = catName;
#        elif (catType == 'leaf'):
#            catMap[catName] = catName
#        else:
#            assert false;
#
#    logging.info('Constructed mapping');
# pprint(catMap);
#
# --- apply mappings
#    ignoredLabelSet = set();
#    for msg in labeledMessageList:
#        labelList = msg[1]
#        newLabelSet = set();
#        for label in labelList:
#            if (label in catMap):
#                newLabelSet.add(catMap[label])
#            else:
#                ignoredLabelSet.add(label);
#        msg[1] = list(newLabelSet);
#
#    logging.info('Ignored labels: %s', str(ignoredLabelSet));
#
#    return labeledMessageList;

def getFullMessagesFromUchaguziMergedCategory(uchaguziJsonPath, uchaguziCategoryJsonPath):
    messageList = getFullMessagesFromJson(uchaguziJsonPath)
    #- map 'Polling station logisitcal issues' to 'Polling Station Logisitcal Issues'
    #- remove duplicate category labels
    for msg in messageList:
        categories = msg['categories']
        for j in range(len(categories)):
            if categories[j] == 'Polling station logisitcal issues':
                categories[j] = 'Polling Station Logistical Issues'
        msg['categories'] = list(set(categories))

    #--- these are selected categories. let's transform
    selectedCategories = [('parent', 'Counting + Results'),
                          ('parent', 'Fear and Tension'),
                          ('parent', 'POSITIVE EVENTS'),
                          ('parent', 'Polling Station Administration'),
                          ('parent', 'Security Issues'),
                          ('parent', 'Staffing Issues'),
                          ('parent', 'Voting Issues'),
                          ('leaf', 'Resolved'),
                          ('leaf', 'Unresolved')]

    categories = loadJsonFromPath(uchaguziCategoryJsonPath)
    categoryByName = dict([(cat['category_title'], cat) for cat in categories])
    categoryById = dict([(cat['id'], cat) for cat in categories])

    #--- create mappings
    catMap = {}
    for selectedCat in selectedCategories:
        catType = selectedCat[0]
        catName = selectedCat[1]
        id = categoryByName[catName]['id']
        if (catType == 'parent'):
            #- find all categories that falls below it, or itself.
            for item in categories:
                if (item['parent_id'] == id or item['id'] == id):
                    catMap[item['category_title']] = catName
        elif (catType == 'leaf'):
            catMap[catName] = catName
        else:
            assert false

    logging.info('Constructed mapping')

    #--- apply mappings
    ignoredLabelSet = set()
    for msg in messageList:
        labelList = msg['categories']
        newLabelSet = set()
        for label in labelList:
            if (label in catMap):
                newLabelSet.add(catMap[label])
            else:
                ignoredLabelSet.add(label)
        msg['categories'] = list(newLabelSet)

    logging.info('Ignored labels: %s', str(ignoredLabelSet))

    return messageList


def loadDatasetWithMappedCategories(dsetJsonPath, mappedCategoryPath):
    #---- read dataset
    messageList = getFullMessagesFromJson(dsetJsonPath)
    for msg in messageList:
        msg['categories'] = list(set(msg['categories']))

    #---- read mappedCategory
    catMap = {}
    with open(mappedCategoryPath, 'rb') as inf:
        csvReader = UnicodeDictReader(inf)
        # headers = csvReader.fieldnames;
        for row in csvReader:
            # json.json.dumps(row)
            engCat = row['Category (English)']
            superCat = row['Super Category']
            assert (superCat is not None and superCat != '')
            catMap[engCat] = superCat

    #---- apply mapping
    for msg in messageList:
        catList = msg['categories']
        newCatSet = set()
        for cat in catList:
            mappedCat = catMap[cat]
            if (mappedCat not in ['Other', '?']):
                newCatSet.add(mappedCat)
        msg['categories'] = list(newCatSet)

    return messageList
