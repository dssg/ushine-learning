import pickle
import jinja2
import json
import random
import time
import operator
import pprint
from flask import session, render_template, request, flash

import dssg.Machine
from dssg.forms import MessageForm
from dssg.webapp import app
from dssg.junutils import *


#
# Routes
#
mac = None
MACHINE_FILENAME = 'machine.data'

@app.route("/")
def main_menu():
    return home()

@app.route("/home", methods=['GET', 'POST'])
def home():

    form = MessageForm()
    g = None

    # Must be a POST request, with message data, having pressed 
    # the "submit" (not "I'm feeling lucky") butto
    if request.method == 'POST' and form.message.data and form.submit.data:
        # User submitted message
        g = guess(form.message.data)
    elif request.method == 'POST' and form.lucky.data:
        # Random message
        g = guess()
    else:
        return render_template('home.html', form=form, form_only=True)
    
    pprint.pprint(g)

    form.message.data=g['actual_message']

    # TODO: Fill the form with the text of the message
    highlighted_message = highlight_entities(g['actual_message'],
                                             g['guessed_entities'])
    
    # Preserve HTML markup, so we can pass <span>s for highlighting text
    highlighted_message = jinja2.Markup(highlighted_message)

    # For now, only show one language (that's all we're guessing)
    guessed_language = g['guessed_language'][0].keys()[0]

    return render_template('home.html', form=form,
                           message=highlighted_message,
                           actual_categories=g['actual_labels'],
                           suggested_categories=g['guessed_labels'],
                           guessed_language=guessed_language,
                           similar_messages=g['guessed_similar_messages'])
    # TODO: Can pass args as a dictionary rather than item by item?

## For now, run training via 
## `demo.py -t`
## to generate the machine.data file

# @app.route("/train")
# def train():
#     new_machine()
#     mac = load_machine()
#     if not mac:
#         return "Error loading machine"

#     # print mac.messages
#     labeledMessageList = getLabeledMessagesFromJson(uchaguziJsonPath)
#     msg = uchaguziJsonPath
    
#     # Train on subset (say 75%) of messages
#     nTrain = int(round(float(len(labeledMessageList))*.75));
#     train = labeledMessageList[0:nTrain]
#     test = labeledMessageList[nTrain:]

#     start = time.time()
#     mac.train(train)
#     duration = time.time() - start
#     print "training time (seconds) = %s" % (duration, )

#     save_machine(mac)

#     return links_menu() + "training with message '%s'" % (msg,) + '<br /><br />' + machine_status()


@app.route("/machine")
def machine():
    mac = load_machine()
    if not mac:
        print "Error loading machine"

    return links_menu() + machine_status()

# @app.route("/guess")
def guess(text = ''):
    # return 'yeah!'
    mac = load_machine()

    if not mac:
        print "Error loading machine"

    if not text:
        labeledMessageList = getFullMessagesFromUchaguziMergedCategory(
            uchaguziJsonPath, uchaguziCategoryJsonPath);

        # Test on remainder of messages (say 25%) of messages
        random.seed(0);  # Just for testing purpose
        random.shuffle(labeledMessageList)
        nTrain = int(round(float(len(labeledMessageList))*.75));
        test = labeledMessageList[nTrain:]
        random.seed()
        m = random.choice(test)
    else:
        # message: user text, categories: none (not labled)
        m = (text, [])

    print m

    actual_message = m['description']
    actual_labels = m['categories']

    # guess language
    language_guess = mac.guess_language(actual_message)

    entities = mac.guess_entities(actual_message)

    g = mac.guess(m[0])

    # TODO: Put this in the machine, write a unit test
    #   - give a specific training set (fake data)
    #   - make a guess and make sure specific categories show up?
    #       or at least right format (list, e.g.)
    def get_top_x(guesses_list = [], num_to_fetch = 5, tag_type = 'categories'):
        """
        :param guesses_list a list of guesses, of form
            [{'categories': { ... }, 'entities': { ... }, ... ]
        as returned by machine
        :param num_to_fetch quantity of categories to 
        """
        output = []
        for x in guesses_list:
            x = x[tag_type]
            sorted_x = sorted(x.iteritems(), key=operator.itemgetter(1))
            sorted_x.reverse()
            output.append(sorted_x[:num_to_fetch])
        return output

    gtext = 'Guessed Labels: <br />'

    for guessed_label in get_top_x(g,5,'categories')[0]:
        guess_format = "%s => %s <br />" % (str(guessed_label[0]), guessed_label[1])
        gtext += guess_format

    # for guessed_label in get_top_x(g,5,'entities')[0]:
    #     guess_format = "%s => %s <br />" % (str(guessed_label[0]), guessed_label[1])
    #     egtext += guess_format

    # print gtext

    similar_messages = g[0]['similar_messages']

    guessed_labels = map(lambda x: x[0], get_top_x(g)[0])

    return {
        'actual_message' : actual_message,
        'actual_labels' : actual_labels,
        'guessed_entities' : entities,
        'guessed_labels' : guessed_labels,
        'guessed_language' : language_guess,
        'guessed_similar_messages' : similar_messages, 
    }


#
# Helpers
#


def links_menu():
    menu = 'MENU' + '<br />'

    return menu + __link('train') + __link('guess') + '<br />'


def __link(text):
    return '<a href="%s">%s</a><br />' % (text, text)


def machine_status():
    return ''

    # mac = load_machine()
    # status = 'MACHINE STATUS' + '<br />'
    # status += str(mac.categories)

    # # if mac:
    # #     status += 'Messages:' + '<br />'

    # #     msgs = ''
    # #     if mac.messages:
    # #         for m in mac.messages:
    # #             msgs += (' - ' + m.text + '<br />')
    # #     else:
    # #         msgs = 'None'

    # #     status += msgs
    # # else:
    # #     status += 'no machine' +  '<br />'

    # return status


def new_machine():
    print "creating new machine"
    mac = Machine.Machine()
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

# remove dups. TODO: remove nested words such as "police" "policeman"
# TODO: html
# Message contains a <span class="entity_location">Location</span> and <span class="entity_other">Another Entity</span>.
# TODO: location vs other

# [('ORGANIZATION', u'Nkumbo Nursery School'),
# ('ORGANIZATION', u'farGeolocation'), ('PERSON', u'Nkumbos'), 
# ('GPE', u'Mwimbi'), ('GPE', u'Maara'), ('ORGANIZATION', u'BOH')]

def highlight_entities(text="", entities=None):
    entities = set(entities)
    for e in entities:
        e_type = e[0]
        e_text = e[1]
        tag = ""
        if e_type == 'GPE': # Tag locations differently
            tag = 'entity_location'
        else:
            tag = 'entity_other'
        text = text.replace(e_text, '<span class="' + tag + '">' + e_text + '</span>')
        # text = text.replace(e_text, tag + e_text + tag)
    return text
