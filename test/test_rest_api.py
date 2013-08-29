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

API_VERSION = '/v1'


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
            API_VERSION + '/language', data=send_json, content_type='application/json')

        return_json = json.loads(rv.data)

        key_lang = 'language'
        key_conf = 'confidence'

        # self.assertIsInstance(
        # json_val, list, 'Should return a list of languages, under "%s" key in
        # json' % (json_key,))

        # self.assertIsInstance(
        #     json_val[0], list, 'Each item is a list')
        #
        self.assertIsInstance(
            return_json, dict, 'Should return a dict')

        self.assertTrue(
            key_lang in return_json, 'Json should have key "%s"' % (key_lang,))

        self.assertTrue(
            key_conf in return_json, 'Json should have key "%s"' % (key_conf,))

        # TODO: Should be unicode, or str?
        actual = type(return_json[key_lang])
        expected = unicode
        self.assertIs(
            actual, expected, '"%s" element should be a %s, instead was %s' % (key_lang, expected, actual))

        self.assertEqual(
            len(return_json[key_lang]), 2, '"%s" element should be a 2-letter language code' % (key_lang,))

        self.assertIs(
            type(return_json[key_conf]), float, 'Second element should be a float, 0-to-1 probability')

        self.assertTrue(return_json[key_conf] >= 0 and return_json[
                        key_conf] <= 1, 'Second element should be in range 0-to-1, because is a probability')

        # TODO: move to non-API tests. belongs in machine tests
        # validity of method belongs in machine tests, while validity of API
        # interface goes here
        self.assertEqual(
            return_json[key_lang], 'en', 'First value should be English')

        # TODO: move to non-API tests, belongs in machine tests
        # expected_language_count = 97  # using langid
        # self.assertEqual(
        # len(json_val), expected_language_count, '%s languages should be
        # returned, but instead got %s' % (expected_language_count,
        # len(json_val)))

    def test_suggest_locations(self):
        text = "My name is Indigo Montoya. I am from the Congo."
        send_json = json.dumps({
            'text': text
        })
        rv = self.app.post(
            API_VERSION + '/locations', data=send_json, content_type='application/json')

        print rv
        print rv.data

        return_json = json.loads(rv.data)
        json_key = 'locations'
        json_val = return_json[json_key]

        key_gsp = "GSP"

        pprint(json_val)

        self.assertIsInstance(
            json_val, dict, 'Should return a dict of location entity types')

        self.assertIsInstance(
            json_val[key_gsp], list, 'Each entity type is composed of a list of items')

        # self.assertIsInstance(
        # json_val, list, 'Should return a list of locations, under "%s" key in
        # json' % (json_key,))

        # self.assertIsInstance(
        #     json_val[0], list, 'Each item is a list')

        # TODO: Should be unicode, or str?
        actual = type(json_val[key_gsp][0])
        expected = unicode
        self.assertIs(
            actual, expected, 'First element should be a %s, instead was %s' % (expected, actual))

        actual = json_val.keys()[0]
        # TODO: fetch from machine.py, so stays in sync?
        expected = ['LOCATION', 'GPE', 'GSP']  # location entity types
        self.assertIn(
            actual, expected, 'Dict key should be a code for location entity type')

        self.assertIs(
            type(json_val[key_gsp][0]), unicode, 'List item should be unicode text for entity name')

        # TODO: move to non-API tests. belongs in machine tests
        self.assertEqual(
            json_val[key_gsp][0], 'Congo', 'First value should be Congo')

        expected_count = 1
        self.assertEqual(
            len(json_val[key_gsp]), expected_count, '%s locations should be returned, but instead got %s' % (expected_count, len(json_val)))

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
