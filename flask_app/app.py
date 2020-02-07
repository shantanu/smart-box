from flask import Flask, request
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Routes
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/data_entry', methods=['POST'])
def enter_data():
    req_data = request.get_json()
    print("Got request")
    box = req_data['box_name']
    print(box)
    channels = req_data['channels']
    print(channels)
    data = req_data['datapoints']
    print(data)
    for timestamp, values in data.items():
        for i in range(len(channels)):
            addData(box, channels[i], timestamp, values[i])


    db.session.commit()
    displayTable(Data)

    return "Data Entered"


# Wrapper Functions

def addData(box, channel, time, value, label=None):
    data = Data(box_name=box, channel_name=channel, time=time, value=value, label=label)
    db.session.add(data)

def displayTable(table):
    print(table.query.all())

# On Start 
def get_env_variable(name):
    try:
        return os.environ[name]
    except KeyError:
        message = "Expected environment variable '{}' not set.".format(name)
        raise Exception(message)

# the values of those depend on your setup
POSTGRES_URL = get_env_variable("POSTGRES_URL")
POSTGRES_USER = get_env_variable("POSTGRES_USER")
POSTGRES_PW = get_env_variable("POSTGRES_PW")
POSTGRES_DB = get_env_variable("POSTGRES_DB")

DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # silence the deprecation warning

db = SQLAlchemy(app)


# Database Models

class Box(db.Model):
    box_name = db.Column(db.String(255), primary_key=True)

    def __repr__(self):
        return '<Box %r>' % self.box_name

class Channel(db.Model):
    channel_name = db.Column(db.String(255), primary_key=True)
    sensor_name = db.Column(db.String(255))

    def __repr__(self):
        return '<Channel {channel_name}, Sensor {sensor_name}>'.format(channel_name=self.channel_name, sensor_name=self.sensor_name)

class Data(db.Model):
    box_name = db.Column(db.String(255), db.ForeignKey('box.box_name'), primary_key=True)
    channel_name = db.Column(db.String(255), db.ForeignKey('channel.channel_name'), primary_key=True)
    time = db.Column(db.DateTime, primary_key=True)
    value = db.Column(db.Float)
    label = db.Column(db.String(255))

    def __repr__(self):
        return '<Data: {}, {}, {}, {}, {}>'.format(self.box_name, 
            self.channel_name, self.time, self.value, self.label)

#addData("box0", "temperature", datetime.now(), 75.29)

displayTable(Data)