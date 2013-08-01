import hashlib
from sqlalchemy import Column, String, Integer, Sequence
from sqlalchemy.orm import relationship

import base_model
from dssg.model import Session

class Deployment(base_model.BaseModel):
    
    __tablename__ = 'deployment'
    
    id = Column(Integer, Sequence('seq_deployment'), primary_key=True)
    name = Column(String)
    url = Column(String, nullable=False)
    url_hash = Column(String, nullable=False, unique=True)
    message_count = Column(Integer)
    report_count = Column(Integer)
    
    # One-to-many relationship definitions
    categories = relationship('Category', backref='deployment',
                              cascade="all, delete, delete-orphan")
    reports = relationship('Report', backref='deployment',
                           cascade="all, delete, delete-orphan")
    messages = relationship('Message', backref='deployment',
                            cascade="all, delete, delete-orphan")
    
    @classmethod
    def by_url(cl, deployment_url):
        """Return the deployment with the given url
        
        :param deployment_url: the url of the deployment
        :type deployment_url: string
        
        :returns: the deployment with the given url or None if there is
            no deployment with that url
        :rtype: dssg.model.Deployment
        
        """
        # Get the MD5 hash of the deployment url
        url_hash = hashlib.md5(deployment_url).hexdigest()
        query = Session.query(Deplyment).filter(Deployment.url_hash==url_hash)
        
        return query.first()
