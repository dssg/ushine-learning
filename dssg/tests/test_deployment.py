from nose.tools import assert_equals

import dssg.tests as _test
from dssg.model import Deployment

class TestDeployment:
    
    @classmethod
    def setup_class(self):
        """Set up preliminary test data"""
        _test.create_deployment('ushine integration', 'http://dssg.ushahididev.com')
        _test.create_deployment('Nigeria budget monitoring',
                                'https://monitoringbudget.crowdmap.com')

    def test_save(self):
        """Tests creation of a new deployment entry"""
        out = _test.create_deployment('uchaguzi 2013',
                                      'http://uchaguzi.co.ke')
        assert_equals(out.url_hash, '87888b7f4d65d4947cde38b99e201544')
    
    def test_by_url(self):
        """Tests finding a deployment by its url"""
        result = Deployment.by_url('http://dssg.ushahididev.com')
        assert_equals(result.url_hash, 'bc2f8da9e34c3fe1ec5fdc2d1fea23c1')

    def test_as_dict(self):
        deployment = Deployment.by_url('https://monitoringbudget.crowdmap.com')
        _dict = deployment.as_dict()
        # assert_equals('categories' in _dict, True)
        # assert_equals('reports' in _dict, True)
        # assert_equals('messages' in _dict, True)
        assert_equals(_dict['name'], 'Nigeria budget monitoring')