import json

from flask import request, jsonify

from dssg.webapp import app
from dssg.Machine import Machine

# g_machine = None


@app.route('/language', methods=['POST'])
def detect_language():
    """Given some text, returns a ranked list of likey natural languages
    the given content is in"""

    mac = Machine()
    try:
        # content = request.json
        content = json.loads(request.data)
        text = content['text']
        g = mac.guess_language(text)
    except:
        print "Failed to load json."
        g = {}
        # TODO: best practices to respond to poorly formatted rest call?

    j = jsonify(g)

    # TODO: Fix formatting of response text. Should return a ranked list
    # instead of dict
    return j


@app.route('/category', methods=['POST'])
def suggest_categories():
    """Given a message/report, suggests the possible categories
    that the message could fall into
    """
    pass


@app.route('/similar', methods=['POST'])
def similar_messages():
    """
    Given text, finds the near duplicate messages.

    input: text
    output: list. made up of tuples of (id, message text).
    [todo: does this only return reports? or unannotated messages, too?
    should be any message for completeness, and then the front-end can decide
    what should be hidden from the user.]
    """
    pass


@app.route('/locations', methods=['POST'])
def suggest_locations():
    """
    Suggest locations in a text string. These might be useful keywords for
    annotators to geolocate.

    input: full message's text [string]
    output: list. each item is a python dictionary:
        - text : the text for the specific entity [string]
        - indices : tuple of (start [int], end [int]) offset where entity is
          located in given full message
        - confidence : probability from 0-to-1 [float]
    """
    mac = Machine()
    try:
        # content = request.json
        content = json.loads(request.data)
        text = content['text']
        g = mac.guess_entities(text)
        # TODO: Should call guess_location instead, which returns locations /
        # GPE only
    except:
        print "Failed to load json."
        g = {}

    j = jsonify(g)

    # TODO: Fix formatting of response text. Should return a ranked list
    # instead of dict
    return j


@app.route('/entities', methods=['POST'])
def extract_entities():
    """Given some text input, identify - besides location - people,
    organisations and other types of entities within the text"""
    mac = Machine()
    try:
        # content = request.json
        content = json.loads(request.data)
        text = content['text']
        g = mac.guess_entities(text)
        # TODO: Should call guess_location instead, which returns locations /
        # GPE only
    except:
        print "Failed to load json."
        g = {}

    j = jsonify(g)

    # TODO: Fix formatting of response text. Should return a ranked list
    # instead of dict
    return j


@app.route('/private_info', methods=['POST'])
def suggest_sensitive_info():
    """
    Suggest personally identifying information (PII) -- such as
    credit card numbers, phone numbers, email, etc --
    from a text string. These are useful for annotators to investigate
    and strip before publicly posting information.

    input: text,
    input: options
        - custom regex for local phone numbers
        - flags or booleans to specify the type of pii (e.g. phone_only)
    output: list of dictionaries:
        - word
        - type (e-mail, phone, ID, person name, etc.)
        - indices (start/end offset in text)
        - confidence [todo: is possible?]
    """
    mac = Machine()
    try:
        # content = request.json
        content = json.loads(request.data)
        text = content['text']
        g = mac.guess_private_info(text)
        # TODO: Should call guess_location instead, which returns locations /
        # GPE only
    except:
        print "Failed to load json."
        g = {}

    j = jsonify(g)

    # TODO: Fix formatting of response text. Should return a ranked list
    # instead of dict
    return j
