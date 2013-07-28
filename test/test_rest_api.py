#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
These test cases focus on the REST API. They aim to enforce:

- consistency in returned JSON object, corresponding to documentation / user expectation.
- poorly formed REST calls are handled properly and consistently.

"""


from dssg.webapp.rest_api import app

import unittest
import json
from pprint import pprint


class TestRestApi(unittest.TestCase):

    #
    # Setup / Teardown
    #

    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    #
    # Tests
    #

    def test_detect_language(self):
        text = "This is a posting in the English language, so the most likely language should be English."
        send_json = json.dumps({
            'text': text
        })
        rv = self.app.post(
            '/language', data=send_json, content_type='application/json')

        return_json = json.loads(rv.data)
        json_key = 'languages'
        json_val = return_json['languages']

        pprint(json_val)

        self.assertIsInstance(
            json_val, list, 'Should return a list of languages, under "%s" key in json' % (json_key,))

        self.assertIsInstance(
            json_val[0], list, 'Each item is a list')

        # TODO: Should be unicode, or str?
        actual = type(json_val[0][0])
        expected = unicode
        self.assertIs(
            actual, expected, 'First element should be a %s, instead was %s' % (expected, actual))

        self.assertEqual(
            len(json_val[0][0]), 2, 'First element should be a 2-letter language code')

        self.assertIs(
            type(json_val[0][1]), float, 'Second element should be a float, 0-to-1 probability')

        # TODO: move to non-API tests. belongs in machine tests
        # validity of method belongs in machine tests, while validity of API
        # interface goes here
        self.assertEqual(
            json_val[0][0], 'en', 'First value should be English')

        # TODO: move to non-API tests, belongs in machine tests
        expected_language_count = 97  # using langid
        self.assertEqual(
            len(json_val), expected_language_count, '%s languages should be returned, but instead got %s' % (expected_language_count, len(json_val)))

    def test_suggest_locations(self):
        text = "My name is Indigo Montoya. I am from the Congo."
        send_json = json.dumps({
            'text': text
        })
        rv = self.app.post(
            '/locations', data=send_json, content_type='application/json')

        print rv
        print rv.data

        return_json = json.loads(rv.data)
        json_key = 'locations'
        json_val = return_json[json_key]

        pprint(json_val)

        self.assertIsInstance(
            json_val, list, 'Should return a list of locations, under "%s" key in json' % (json_key,))

        self.assertIsInstance(
            json_val[0], list, 'Each item is a list')

        # TODO: Should be unicode, or str?
        actual = type(json_val[0][0])
        expected = unicode
        self.assertIs(
            actual, expected, 'First element should be a %s, instead was %s' % (expected, actual))

        actual = json_val[0][0]
        expected = ['LOCATION', 'GPE', 'GSP']
            # location entity types; TODO: fetch from machine.py, so stays in
            # sync?
        self.assertIn(
            actual, expected, 'First element should be a code for location entity type')

        self.assertIs(
            type(json_val[0][1]), unicode, 'Second element should be unicode entity text')

        # TODO: move to non-API tests. belongs in machine tests
        self.assertEqual(
            json_val[0][1], 'Congo', 'First value should be Congo')

        expected_count = 1
        self.assertEqual(
            len(json_val), expected_count, '%s locations should be returned, but instead got %s' % (expected_count, len(json_val)))

    # TODO: etc... to enforce all REST endpoints
    def test_suggest_sensitive_info(self):
        pass

    def test_extract_entities(self):
        pass

    def test_suggest_categories(self):
        pass

    def test_similar_messages(self):
        pass

if __name__ == '__main__':
    unittest.main()
