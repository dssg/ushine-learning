from sqlalchemy import Column, ForeignKey, Integer, Table

from dssg.model import Base, Session


class BaseModel(Base):
    """Base class for all mapped classes"""
    
    def __init__(self, **kwargs):
        for k,v in kwargs:
            setattr(self, k, v)
    
    @classmethod
    def by_id(cls, id):
        """Load and return by the primary key"""
        return Session.query(cls).get(id)

    def create(self):
        """Saves the current object in the database"""
        Session.add(self)
        Session.commit()
        
    def delete(self):
        """Deletes current object from the database"""
        Session.delete(self)
        Session.commit()