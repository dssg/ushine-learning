#!/usr/bin/python
# -*- coding: utf-8 -*-

from nose.tools import *


def setup():
    print "setup!"


def teardown():
    print "tear down!"


def test_canRunTest():
    assert (True is True)
