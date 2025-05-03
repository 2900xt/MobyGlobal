from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # User preferences
    notifications_enabled = db.Column(db.Boolean, default=True)
    
    # Relationships
    subscriptions = db.relationship('Subscription', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    notifications = db.relationship('Notification', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Region(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.String(256))
    center_lat = db.Column(db.Float, nullable=False)
    center_lng = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Float, default=100.0)  # Radius in kilometers
    
    # Relationships
    subscriptions = db.relationship('Subscription', backref='region', lazy='dynamic', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Region {self.name}>'

class Sensor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    location = db.Column(db.String(128), nullable=False)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    region = db.relationship('Region', backref='sensors')
    subscriptions = db.relationship('Subscription', backref='sensor', lazy='dynamic', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Sensor {self.name}>'

class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    sensor_id = db.Column(db.Integer, db.ForeignKey('sensor.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # A subscription can be for a region or a specific sensor
    # At least one of region_id or sensor_id must be set
    
    def __repr__(self):
        if self.region_id:
            return f'<Subscription User:{self.user_id} Region:{self.region_id}>'
        else:
            return f'<Subscription User:{self.user_id} Sensor:{self.sensor_id}>'

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
    
    # Optional reference to a sensor or region that triggered the notification
    sensor_id = db.Column(db.Integer, db.ForeignKey('sensor.id'))
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    
    # Relationships
    sensor = db.relationship('Sensor', backref='notifications')
    region = db.relationship('Region', backref='notifications')
    
    def __repr__(self):
        return f'<Notification {self.id} for User:{self.user_id}>'
