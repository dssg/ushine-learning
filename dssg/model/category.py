from dssg import db
import base_model

class Category(base_model.BaseModel, db.Model):
    """Mapping for the category table"""
    
    __tablename__ = 'category'
    
    id = db.Column(db.Integer, db.Sequence('seq_category_id'), primary_key=True)
    deployment_id = db.Column(db.Integer, db.ForeignKey('deployment.id'))
    origin_category_id = db.Column(db.Integer, nullable=False)
    origin_parent_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String, nullable=False)
        