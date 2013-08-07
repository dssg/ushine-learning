from dssg import db

class BaseModel:
    """Base class for all mapped classes"""
    
    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
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