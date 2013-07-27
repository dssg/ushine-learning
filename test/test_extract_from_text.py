#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Verify private data stripping functionality
"""

from nose.tools import *
from dssg.Machine import Machine


#
# Setup / Teardown
#

def setup():
    print "setup!"


def teardown():
    print "tear down!"

#
# Tests
#


def test_canExtractUrls():
    text = "Awesome post talking about http://www.mysite.com which is full of private info!"
    extract_fn = Machine._extract_urls
    expected = [('URL', 'http://www.mysite.com')]
    _helper_canExtractX(text, extract_fn, expected)


def test_canExtractEntities():
    text = "My name is Indigo Montoya. I am from the Congo."
    extract_fn = Machine._extract_entities
    expected = [('PERSON', 'Indigo Montoya'), ('GSP', 'Congo')]
    _helper_canExtractX(text, extract_fn, expected)


def test_canExtractIds():
    text = "Oh my gosh I accidentally included my credit card number 14320099 and passport P123411."
    extract_fn = Machine._extract_ids
    expected = [('ID', '14320099'), ('ID', 'P123411')]
    _helper_canExtractX(text, extract_fn, expected)


def test_canExtractUsernames():
    text = "RT best tweet evarrrr @123fake @justinbieber @BarackObama."
    extract_fn = Machine._extract_usernames
    expected = [('TWITTER', '@justinbieber'), (
        'TWITTER', '@BarackObama'), ('TWITTER', '@123fake')]
    _helper_canExtractX(text, extract_fn, expected)


def test_canExtractEmails():
    text = "Hello my email is fakeperson@example.com and I am here."
    extract_fn = Machine._extract_emails
    expected = [('EMAIL', 'fakeperson@example.com')]
    _helper_canExtractX(text, extract_fn, expected)


def test_canExtractPhones():
    text = "This is my phone number 555-555-3333!"
    extract_fn = Machine._extract_phones
    expected = [('PHONE', '555-555-3333')]
    _helper_canExtractX(text, extract_fn, expected)

#
# Helpers
#


def _helper_canExtractX(text='', extract_fn=None, expected=None):
    actual = extract_fn(text)

    assert(set(expected) == set(
        actual)), "set(expected) != set(actual).\n set(expected): %s, set(actual): %s" % (set(expected), set(actual))
    assert(len(expected) == len(
        actual)), "len(expected) != len(actual).\n len(expected): %s, actual: %s" % (len(expected), len(actual))
