from sqlalchemy import Column, Integer, String, Sequence, ForeginKey

import base_model

class Category(base_model.BaseModel):
    """Mapping for the category table"""
    
    __tablename__ = 'category'
    
    id = Column(Integer, Sequence('seq_category_id'), primary_key=True)
    deployment_id = Column(Integer, ForeginKey('deployment.id'))
    origin_category_id = Column(Integer, nullable=False)
    origin_parent_id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
        