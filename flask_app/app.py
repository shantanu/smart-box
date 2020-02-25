from flask import Flask, request, redirect
import os
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc 
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go


server = Flask("SmartBox Companion App")

# ================== ON START DATABASE CONNECTION ======================
POSTGRES_URL = "localhost:5432"
POSTGRES_USER = "postgres"
POSTGRES_PW = "smartbox"
POSTGRES_DB = "smartbox"

DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(
    user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)

server.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
# silence the deprecation warning
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

db = SQLAlchemy(server)

# ===================== HOMEPAGE REDIRECT =======================
@server.route('/')
def hello_world():
    # TODO: simply reroute this to /smartdash
    return redirect('smartdash')


# =============== DATA ENTRY AND RETRIEVAL REST API ====================

@server.route('/data_entry', methods=['POST'])
def enter_data():
    req_data = request.get_json()
    print("Got request")
    box = req_data['box_name']
    print(box)
    channels = req_data['channel_names']
    #print(channels)
    data = req_data['datapoints']
    #print(data)

    for timestamp, values in data.items():
        ts = datetime.utcfromtimestamp(int(float(timestamp))).strftime('%Y-%m-%d %H:%M:%S')
        value_floats = list(map(float, values.split(",")))
        for i in range(len(channels)):
            addData(box, channels[i], ts, value_floats[i])


    db.session.commit()
    displayTable(Data)
    print("Data Entered")

    return "Data Entered"

@server.route('/check_box', methods=['POST'])
def check_box():
    # box will first send boxname and all channels and associated 
    # sensors. This will ensure that it's all in the database
    # otherwise put.
    print("Check box!")
    req_data = request.get_json()
    #print(req_data)
    box = req_data['box_name']
    print(box)

    box_missing = Box.query.filter_by(box_name=box).first()
    if box_missing is None:
        print("Box missing!", box)
        new_box = Box(box_name=box)
        db.session.add(new_box)

    
    #print(req_data['channel_names'])
    channels = {x[0]:x[1] for x in req_data['channel_names']}
    print(channels)
    for channel, sensor in channels.items():
        channel_missing = Channel.query.filter_by(channel_name=channel).first()
        if channel_missing is None:
            print("Channel missing!", channel)
            new_channel = Channel(channel_name=channel, sensor_name=sensor)
            db.session.add(new_channel)

    db.session.commit()
    print("Box Validated")
    return "Box Validated"


@server.route("/get_data", methods=['GET'])
def get_data_request():
    box = request.args.get('box_name')
    start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
    
    result = get_data(box, start_time, end_time)
    result = [(row[0], row[1], row[2].strftime('%Y-%m-%d %H:%M:%S'), row[3], row[4])
                for row in result]

    return json.dumps(result)

@server.route("/get_box_channels", methods=['GET'])
def get_box_channels_request():
    box = request.args.get('box_name')
    
    result = get_box_channels(box)
    print(result)

    return json.dumps(result)

@server.route("/get_box_names", methods=['GET'])
def get_box_names_request():
    result = get_box_names()

    print(result)

    # TODO: return this in a post request
    return json.dumps(result)

# =========== WRAPPER FUNCTIONS FOR DATABASE QUERIES ===================

def addData(box, channel, time, value, label=None):
    data = Data(box_name=box, channel_name=channel, time=time, value=value, label=label)
    db.session.add(data)

def displayTable(table):
    print(table.query.all())


# start_time and end_time are in python datetime types
# returns a list of tuples (box_name, channel_name, datetime (python format), 
# value, label)
# note: you must parse the datetime object before printing!
def get_data(box, start_time, end_time):
    query = "SELECT * FROM Data d WHERE d.box_name = '{}' and d.time BETWEEN '{}' AND '{}';".format(box, start_time, end_time)
    result = db.engine.execute(text(query))
    return [row for row in result]


# returns a list of channel names for a given box
# list of strings
def get_box_channels(box):
    query = "SELECT DISTINCT channel_name FROM data where box_name = '{}'".format(box)
    print(query)
    result = db.engine.execute(text(query))

    return [channel[0] for channel in result]
    

# returns list of strings of all box_names in database
def get_box_names():
    query = "SELECT * from Box"
    print(query)
    result = db.engine.execute(text(query))

    return [box[0] for box in result]


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






# ========================= FRONTEND DASH APP =============================
smartdash = dash.Dash(name="smartdash", server=server, 
                        url_base_pathname='/smartdash/',
                        external_stylesheets=[dbc.themes.BOOTSTRAP])


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        'overflowY': 'scroll',
        'height': '10pc' 
    },
    '40vh': {
        'height': '40vh', 
        'display': 'block'
    },
    'disappear': {
        'display': 'none',
    },


}




# ========================= LAYOUT ==========================================

# represents the URL bar, doesn't render anything
location = dcc.Location(id='url', refresh=False)

# may have to remove some of these
hidden_vars = html.Div([
    html.Div("/", id="prev-url", style=styles['disappear']),
    html.Div("False", id="paused", style=styles['disappear']),
    html.Div("False", id="live", style=styles['disappear']),
    html.Div("0", id="label-submit-count", style=styles['disappear'])
])

# sticky navbar at top
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ]
        )
    ],
    brand='CyPhy Lab Active Learning Web App',
    brand_href='#',
    sticky='top',
)


dynamic_graphs_layout = [
    dbc.Row([
        html.H1("Please pick a graph from left.", id="collection-title"),
        #dbc.Button("Pause Live Database", id="live-button", style=styles['disappear'], disabled=True),
        dcc.Interval(
            id='interval-component',
            interval=15*1000, # in milliseconds
            n_intervals=0
        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='subplots-graph',
                style={
                    'width': '100%',
                    'height': '80vh'
                }
            )
        ],
        md=8),
        dbc.Col([
            dbc.Row([
                html.H3("Selected Data - Top Graph Only!"),
                html.Pre(id='selected-data', style=styles['pre'], children="Please Select Data by dragging a box on the graph!"),
                dcc.Input(id="label", type="text", placeholder='Activity Label'),
                html.Br(),
                dbc.Button("Submit Label", id="label-submit", color="info", className="mr-1"),
            ],
            style=styles['40vh']),

            html.H2("Here are the Labels in the database:"),
            dbc.ListGroup(
                children=[],
                id="database-labels"
            )
        ],
        md=4)
        
    ])
]



graph_select_form = dbc.Form([
    dbc.FormGroup(
        [
            dbc.Label("Box Name", html_for="box_dropdown"),
            dcc.Dropdown(
                id="box_dropdown",
                options = [
                    {"label": box, "value": box} 
                        for box in get_box_names()
                ],
            )
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Start Time", html_for="start_time"),
            dbc.Input(
                id="start_time",
                placeholder="YYYY-MM-DD H:M:S"
            )
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("End Time", html_for="end_time"),
            dbc.Input(
                id="end_time",
                placeholder="YYYY-MM-DD H:M:S"
            )
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Channel Name", html_for="Channel_dropdown"),
            dcc.Dropdown(
                id="channel_dropdown",
                multi=True
            )
        ]
    ),
    dbc.Button(id="graph-form-submit", children="Graph It!")
])

body = dbc.Container(
    [
        dbc.Row([
            # nav with pills
            dbc.Col(   
                children=[
                    graph_select_form,
                ],
                md=3,
            ),

            dbc.Col( 
                children=dynamic_graphs_layout
            )
        ])
    ],
    className="mt-4",
    style={
            'margin': '0 auto',
            'padding': '0',
    }
)


smartdash.layout = html.Div([navbar, body])
#smartdash.layout = html.Div([graph_select_form])


@smartdash.callback([Output('channel_dropdown', 'options')],
                    [Input('box_dropdown', 'value')])
def update_box_channels(box):
    # print("update box channels")
    # print(box)
    # print(selected)
    # print(options)
    # if selected != None and "All Channels" in selected:
    #     return [[{"label": "All Channels", "value": "All Channels"}]]
    # elif selected != None and selected != []:
    #     x = options.remove({"label": "All Channels", "value": "All Channels"})
    #     return [x]
    #if box == None:
    #    raise PreventUpdate

    channels = ["All Channels"]  + get_box_channels(box)
    print(channels)
    return [[{"label": channel, "value": channel} 
                        for channel in channels]]



@smartdash.callback([Output('subplots-graph', 'figure')],
    [Input('graph-form-submit', 'n_clicks')],
    [State('box_dropdown', 'value'),
     State('start_time', 'value'),
     State('end_time', 'value'),
     State('channel_dropdown', 'value')])

def update_graphs(n, box, start_time, end_time, channels):
    if n == None:
        raise PreventUpdate
    # list of tuples 
    # (box, channel, timestamp, value, label)
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    
    data = get_data(box, start_time, end_time)
    df = pd.DataFrame(data, columns=['box_name', 
        'channel_name', 'timestamp', 'value', 'label'])
    
    df = df.drop(axis=1, columns=["box_name", 'label'])

    # keep only the channels we want
    if "All Channels" in channels:
        channels = list(df['channel_name'].unique())

    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True,
                subplot_titles=channels)
    
    for i, channel in enumerate(channels):
        print(channel)
        chan_data = df[df['channel_name'] == channel].sort_values(by='timestamp')
        fig.append_trace(go.Scatter(
            x = chan_data['timestamp'].tolist(),
            y = chan_data['value'].tolist()
        ), row=i+1, col=1)
    
    fig.update_layout(showlegend=False, 
        xaxis= dict(
            showspikes = True,
            spikemode  = 'toaxis+across',
            spikesnap = 'cursor',
            spikedash = 'solid'
        ),
        dragmode = "select"
    ) 
    return [fig]

