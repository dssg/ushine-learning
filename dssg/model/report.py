from sqlalchemy.ext.associationproxy import association_proxy

import base_model
from dssg import db

class Report(base_model.BaseModel, db.Model):    
    __tablename__ = 'report'    
    id = db.Column(db.Integer, db.Sequence('seq_report'), primary_key=True)
    deployment_id = db.Column(db.Integer, db.ForeignKey('deployment.id'))
    origin_report_id = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    
    # Association proxy of "report_categories" collection
    # to "categories" attribute
    categories = association_proxy('report_categories', 'category')
                                 
class ReportCategory(base_model.BaseModel, db.Model):
    __tablename__ = 'report_category'
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'),
                          primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'),
                            primary_key=True)

    # bi-directional attribute/collection of report/report_categories
    report = db.relationship(Report,
                             backref=db.backref('report_categories',
                                                cascade='all, delete-orphan'))

    # Reference to the category object
    category = db.relationship("Category")
