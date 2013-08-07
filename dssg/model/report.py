from dssg import db
import base_model

# Association table
report_categories = db.Table('report_category',
    db.Column('report_id', db.Integer, db.ForeignKey('report.id')),
    db.Column('category_id', db.Integer, db.ForeignKey('category.id')))

class Report(base_model.BaseModel, db.Model):
    
    __tablename__ = 'report'
    
    id = db.Column(db.Integer, db.Sequence('seq_report'), primary_key=True)
    deployment_id = db.Column(db.Integer, db.ForeignKey('deployment.id'))
    origin_report_id = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=False)
    title = db.Column(db.String, nullable=False)
    
    # Many-to-many relationship Report<-->Category
    categories = db.relationship('Category', secondary=report_categories,
                              backref='report')
    