#!/usr/bin/python
# -*- coding: utf-8 -*-

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


def test_canCreateMachine():
    mac = Machine()
    assert (mac != None)


def test_canGuessLanguage():
    mac = Machine()
    text = "Hello world this is definitely some English text"
    g = mac.guess_language(text)
    assert (g != None)


def test_canGuessEntities():
    # Note: test takes a few seconds to run - loading NTLK is slow
    mac = Machine()
    text = "The United States is a country. Thomas Jefferson was a president. This is definitely Lower Wacker Drive."
    g = mac.guess_entities(text)
    assert (g != None)


def test_canStripPrivateInfo():
    mac = Machine()
    # TODO: Change 'text' to have all the types of private info, not just URL.
    text = "This post talks about http://www.mysite.com which is full of private info!"

    actual = mac.guess_private_info(text)
    expected = [('URL', 'http://www.mysite.com')]

    assert(set(expected) == set(
        actual)), "set(expected) != set(actual).\n set(expected): %s, set(actual): %s" % (set(expected), set(actual))
    assert(len(expected) == len(
        actual)), "len(expected) != len(actual).\n len(expected): %s, actual: %s" % (len(expected), len(actual))

#
# Helpers
#
