import csv, codecs, cStringIO
import os, sys
import json
import copy
import logging
import cPickle as pickle
from datetime import datetime

FORMAT = '%(levelname)s: %(message)s';
logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

uchaguziPath = 'data/processed/uchaguzi/uchaguzi-message_with_category_list.csv'
uchaguziJsonPath = 'data/processed/uchaguzi_new.json'
uchaguziCategoryJsonPath = 'data/processed/uchaguzi_new_categories.json'

################################################################################
# Pickle
################################################################################

def loadPickle(fileName):
    return load_pickle(fileName)

def load_pickle(fileName):
  """ load a pickle file. Assumes that it has one dictionary object that points to
 many other variables."""
  with open(fileName, 'rb') as f:
    varDic = pickle.load(f);
  return varDic;

def savePickle(var, fileName, protocol=0):
  f = open(fileName, 'wb');
  pickle.dump(var, f, protocol=protocol);
  f.close();

################################################################################
# for CSV
################################################################################
class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def UnicodeDictReader(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield dict([(key, unicode(value, 'utf-8')) for key, value in row.iteritems()])

################################################################################
# utiliti functions
################################################################################


def tic():
    """
    equivalent to Matlab's tic. It start measuring time.
    returns handle of the time start point.
    """
    global gStartTime
    gStartTime = datetime.utcnow();
    return gStartTime

def toc(prev=None):
    """
    get a timestamp in seconds. Time interval is from previous call of tic() to current call of toc().
    You can optionally specify the handle of the time ending point.
    """
    if prev==None: prev = gStartTime;
    return (datetime.utcnow() - prev).total_seconds();

def unicodeToAscii(unicodeStr):
  s = copy.copy(unicodeStr)
  while True:
    try:
      ret = codecs.encode(s, 'ascii');
      return ret;
    except UnicodeEncodeError as e:
      s = s.replace(e.object[e.start:e.end], ' ');
  return None;
  
def loadIncidentListFromCsv(path):
    """
    returns a list of incidents. An incident is [incidentId, message, categoryIdList, categoryTitleList].
    However, the categories may have duplicates. Duplicates are not meaningful and should be
    removed, but I simply did not remove them in this function.
    """
    incidentList = [];
    with open(path,'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        bFirstLine = True;
        for row in reader:
            if (bFirstLine): # skip the first line
                bFirstLine=False;
                continue;
            assert(len(row) == 4);
            incidentId = int(row[0]);
            message = row[1];
            ### Note that there are duplicates.
            categoryIdList = map(lambda x: int(x), row[2].split(','))
            categoryTitleList = map(lambda x: x.strip(), row[3].split(','))
            assert(len(categoryIdList) == len(categoryTitleList));
   
            incidentList.append([incidentId, message, categoryIdList, categoryTitleList])
    return incidentList

def getFullMessagesFromJson(path):
    with open(path,'r') as fp:
        data = json.load(fp)
        messageList = []
        keyList = sorted(data.keys()) #- sort by keys.
        for k in keyList:
            v = data[k];
            v[u'id'] = k;
            messageList.append(v);
        return messageList;

def countTruth(boolFunc, aList):
    return len(filter(boolFunc, aList));

def isAllNumbers(aStr):
    aStr = aStr.strip();
    if (len(aStr) == countTruth(lambda x: x>='0' and x <= '9', aStr)):
        return True;
    else:
        return False;

def loadJsonFromPath(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data;

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
    selectedCategories = [('parent', 'Counting + Results'),\
                          ('parent', 'Fear and Tension'),\
                          ('parent', 'POSITIVE EVENTS'),\
                          ('parent', 'Polling Station Administration'),\
                          ('parent', 'Security Issues'),\
                          ('parent', 'Staffing Issues'),\
                          ('parent', 'Voting Issues'),\
                          ('leaf',   'Resolved'),\
                          ('leaf',   'Unresolved')]

    categories = loadJsonFromPath(uchaguziCategoryJsonPath);
    categoryByName = dict( [(cat['category_title'], cat) for cat in categories])
    categoryById = dict( [(cat['id'], cat) for cat in categories]) 

    #--- create mappings
    catMap = {}
    for selectedCat in selectedCategories:
        catType = selectedCat[0]; catName = selectedCat[1];
        id = categoryByName[catName]['id'];
        if (catType == 'parent'):
            #- find all categories that falls below it, or itself.
            for item in categories:
                if (item['parent_id'] == id or item['id'] == id):
                    catMap[item['category_title']] = catName;
        elif (catType == 'leaf'):
            catMap[catName] = catName
        else:
            assert false;   
    
    logging.info('Constructed mapping');

    #--- apply mappings
    ignoredLabelSet = set();
    for msg in messageList:
        labelList = msg['categories']
        newLabelSet = set();
        for label in labelList:
            if (label in catMap):
                newLabelSet.add(catMap[label])
            else:
                ignoredLabelSet.add(label);
        msg['categories'] = list(newLabelSet);

    logging.info('Ignored labels: %s', str(ignoredLabelSet));

    return messageList;

def loadDatasetWithMappedCategories(dsetJsonPath, mappedCategoryPath):
    #---- read dataset
    messageList = getFullMessagesFromJson(dsetJsonPath)
    for msg in messageList:
        msg['categories'] = list(set(msg['categories']));

    #---- read mappedCategory
    catMap = {};
    with open(mappedCategoryPath, 'rb') as inf:
        csvReader = UnicodeDictReader(inf);
        #headers = csvReader.fieldnames;
        for row in csvReader:
            #json.json.dumps(row)
            engCat = row['Category (English)']
            superCat = row['Super Category']
            assert ( superCat != None and superCat != '');
            catMap[engCat] = superCat;

    #---- apply mapping
    for msg in messageList:
        catList = msg['categories']
        newCatSet = set();
        for cat in catList:
            mappedCat = catMap[cat];
            if (mappedCat not in ['Other', '?']):
                newCatSet.add(mappedCat);
        msg['categories'] = list(newCatSet)

    return messageList;

