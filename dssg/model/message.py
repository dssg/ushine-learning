from dssg import db
import base_model


class Message(base_model.BaseModel, db.Model):

    __tablename__ = 'message'

    id = db.Column(db.Integer, db.Sequence('seq_message'), primary_key=True)
    deployment_id = db.Column(db.Integer, db.ForeignKey('deployment.id'))
    origin_message_id = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    simhash = db.Column(db.String(64), nullable=False)
