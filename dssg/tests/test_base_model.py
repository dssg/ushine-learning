from nose.tools import assert_equals
from sqlalchemy import func

from dssg import db
from dssg.model import Category

def test_bulk_create():
    """Tests creation of several items in a single batch"""
    categories = []
    categories.append(Category(title='category 1',
                               origin_category_id=1,
                               origin_parent_id=0))
    categories.append(Category(title='category 2', 
                               origin_category_id=2,
                               origin_parent_id=0))
    categories.append(Category(title='category 2 child',
                               origin_category_id=3,
                               origin_parent_id=2))
    Category.create_all(categories)
    
    count = db.session.query(func.count('*')).select_from(Category).scalar()
    assert_equals(count, 3);
