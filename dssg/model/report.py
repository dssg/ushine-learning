from sqlalchemy import Column, ForeignKey, Integer, Sequence, Table, Text
from sqlalchemy.orm import relationship

import base_model
from dssg.model import Base

# Association table
report_categories = Table('report_category', Base.metadata,
    Column('report_id', Integer, ForeignKey('report.id')),
    Column('category_id', Integer, ForeignKey('category.id')))

class Report(base_model.BaseModel):
    
    __tablename__ = 'report'
    
    id = Column(Integer, Sequence('seq_report'), primary_key=True)
    deployment_id = Column(Integer, ForeignKey('deployment.id'))
    origin_report_id = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    
    # Many-to-many relationship Report<-->Category
    categories = relationship('Category', secondary=report_categories,
                              backref='report')
    