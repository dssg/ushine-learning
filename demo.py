#!/usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Change pickle to cpickle. pickle in binary representation.

import dssg.Machine
import cPickle as pickle
import json
from dssg.junutils import *
import random
import time
import operator
import argparse
from pprint import pprint

# Show logging for training of classifer
import sys
import logging
from functools import reduce
FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT, stream=sys.stderr, level=logging.INFO)

MACHINE_FILENAME = 'machine.data'
mac = None
# labeledMessageList = getLabeledMessagesFromJson(uchaguziJsonPath)[:-1]
labeledMessageList = getFullMessagesFromUchaguziMergedCategory(
    uchaguziJsonPath, uchaguziCategoryJsonPath)

# Train on subset (say 75%) of messages
random.seed(0)
# Just for testing purpose
random.shuffle(labeledMessageList)
nTrain = int(round(float(len(labeledMessageList)) * .75))
# nTrain = 300
trainingSet = labeledMessageList[0:nTrain]
validationSet = labeledMessageList[nTrain:]


def main():
    parser = argparse.ArgumentParser(
        description='Demo of DSSG-Ushahidi machine learning')

    parser.add_argument('-t', '--train', action='store_true', default=False,
                        dest='boolean_train',
                        help='Train the machine with a dataset')
                        # TODO: Allow passing "which" dataset as a parameter...
                        # name the outputted machine file appropriately
    parser.add_argument('-g', '--guess', action='store_true', default=False,
                        dest='boolean_guess',
                        help='Select a random item from the validation set, then show result of guessing with the trained model')
    parser.add_argument(
        '-m', '--message', help='Custom message to guess on', required=False, default=None)

    args = vars(parser.parse_args())

    if args['boolean_train']:
        print train()

    if args['boolean_guess']:
        pprint(guess())

    if args['message']:
        print args['message']
        pprint(guess(args['message']))

    # check if any arguments have a value which is not None or False
    arguments_exist = reduce(lambda x, y: x or y, args.values())
    if not arguments_exist:
        print "use -h to view help"


def train():
    new_machine()
    mac = load_machine()
    if not mac:
        return "Error loading machine"

    start = time.time()

    print len(trainingSet)

    mac.train(trainingSet)
    duration = time.time() - start
    print "training time (seconds) = %s" % (duration, )

    save_machine(mac)


# TODO: allow passing any message_id, to check its output
def guess(text=None):
    mac = load_machine()

    if not mac:
        print "Error loading machine"

    if not text:
        random.seed()
        # - use system time to set seed
        m = random.choice(validationSet)
        # pprint(m)
        actual_message = m['description']
        actual_labels = m['categories']
    else:
        actual_message = text
        actual_labels = None

    print
    print "Message text..."
    print actual_message

    print
    print "Guessing categories and similar messages..."
    g = mac.guess([m])
    pprint(g)

    print
    print "Guessing Language..."
    language_guess = mac.guess_language(actual_message)
    pprint(language_guess)

    print
    print "Guessing entities..."
    ent_text = mac.guess_entities(actual_message)
    pprint(ent_text)


#
# Helpers
#

def new_machine():
    print "creating new machine"
    mac = dssg.Machine.Machine()
    mac = save_machine(mac)


def load_machine():
    global mac

    print "loading machine from", MACHINE_FILENAME
    start = time.time()

    if not mac:
        try:
            f = open(MACHINE_FILENAME, 'r')
            mac = pickle.load(f)

        except Exception:
            print "failed to load data from %s" % (MACHINE_FILENAME,)

    duration = time.time() - start
    print "load time (seconds) = %s" % (duration, )

    return mac


def save_machine(mac=None):
    f = open(MACHINE_FILENAME, 'w')
    pickle.dump(mac, f)
    f.close()

    return mac

if __name__ == "__main__":
    main()
