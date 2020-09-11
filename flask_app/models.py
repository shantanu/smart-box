from flask.ext.sqlalchemy import SQLAlchemy
from app import server
db = SQLAlchemy(server)

# ==================== DATABASE MODELS ===============================

class Box(db.Model):
    box_name = db.Column(db.String(255), primary_key=True)

    def __repr__(self):
        return '<Box %r>' % self.box_name

class Channel(db.Model):
    channel_name = db.Column(db.String(255), primary_key=True)
    sensor_name = db.Column(db.String(255))

    def __repr__(self):
        return '<Channel {channel_name}, Sensor {sensor_name}>'.format(
            channel_name=self.channel_name, sensor_name=self.sensor_name)

class Data(db.Model):
    box_name = db.Column(db.String(255), db.ForeignKey('box.box_name'), 
                        primary_key=True)
    channel_name = db.Column(db.String(255), 
        db.ForeignKey('channel.channel_name'), primary_key=True)
    time = db.Column(db.DateTime, primary_key=True)
    value = db.Column(db.Float)
    label = db.Column(db.String(255))

    def __repr__(self):
        return '<Data: {}, {}, {}, {}, {}>'.format(self.box_name, 
            self.channel_name, self.time, self.value, self.label)


class Picture(db.Model):
    box_name = db.Column(db.String(255), db.ForeignKey('box.box_name'),
                        primary_key=True)
    time = db.Column(db.DateTime, primary_key=True)
    picture = db.Column(db.LargeBinary)

    def __repr__(self):
        return '<Picture: {}, {}, {}>'.format(self.box_name, 
            self.time, len(self.picture))
