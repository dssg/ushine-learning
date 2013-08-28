import datetime

from sqlalchemy import orm

from dssg import db


class BaseModel:

    """Base class for all mapped classes"""

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @classmethod
    def by_id(cls, id):
        """Load and return by the primary key"""
        return db.session.query(cls).get(id)

    def create(self):
        """Saves the current object in the database"""
        db.session.add(self)
        db.session.commit()

    def delete(self):
        """Deletes current object from the database"""
        db.session.delete(self)
        db.session.commit()

    def as_dict(self):
        """Returns a dictionary representation of a database
        table row

        :rtype: dict
        """
        _dict = {}
        table = orm.class_mapper(self.__class__).mapped_table
        for col in table.c:
            val = getattr(self, col.name)
            if isinstance(val, datetime.date):
                val = str(val)
            if isinstance(val, datetime.datetime):
                val = val.isoformat()
            _dict[col.name] = val
        return _dict

    @classmethod
    def create_all(cls, entries=[]):
        """Saves a list of objects in bulk

        :param entries: the list of objects to be saved
        """
        for row in entries:
            db.session.add(row)
        if len(entries) > 0:
            db.session.commit()
