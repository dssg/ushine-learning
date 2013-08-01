from sqlalchemy import Column, ForeignKey, Integer, Sequence, Text

import base_model

class Message(base_model.BaseModel):
    
    __tablename__ = 'message'
    
    id = Column(Integer, Sequence('seq_message'), primary_key=True)
    deployment_id = Column(Integer, ForeignKey('deployment.id'))
    content = Column(Text, nullable=False)