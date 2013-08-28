import hashlib

from dssg import db
import base_model


class Deployment(base_model.BaseModel, db.Model):

    __tablename__ = 'deployment'

    id = db.Column(db.Integer, db.Sequence('seq_deployment'), primary_key=True)
    name = db.Column(db.String(50))
    url = db.Column(db.String(100), nullable=False)
    url_hash = db.Column(db.String(32), nullable=False, unique=True)
    message_count = db.Column(db.Integer)
    report_count = db.Column(db.Integer)

    # One-to-many relationship definitions
    categories = db.relationship('Category', backref='deployment',
                                 cascade="all, delete, delete-orphan")
    reports = db.relationship('Report', backref='deployment',
                              cascade="all, delete, delete-orphan")
    messages = db.relationship('Message', backref='deployment',
                               cascade="all, delete, delete-orphan")

    def save(self):
        self.url_hash = hashlib.md5(self.url).hexdigest()
        db.session.add(self)
        db.session.commit()

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
        return Deployment.query.filter_by(url_hash=url_hash).first()
