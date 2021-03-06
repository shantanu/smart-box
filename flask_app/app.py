from flask import Flask, request, redirect
from flask_migrate import Migrate
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
#import dash_gif_component as Gif
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
from datetime import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image
import glob

import data_processing
#import dae_cpd

import tsfresh

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling


# class BaseConfig(object):
#     DEBUG = False
#     TESTING = False
#     #SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
#     SQLALCHEMY_DATABASE_URI = 'postgres://zpsubtcgyhmhue:1558b0987b0ec9fa461f2072c176d2712e0498d7541cc965a5f0bb007766e18e@ec2-3-226-231-4.compute-1.amazonaws.com:5432/d88km9ic1ahkbe'
#     SQLALCHEMY_TRACK_MODIFICATIONS = False

# class TestingConfig(BaseConfig):
#     DEBUG = False
#     TESTING = True
#     SQLALCHEMY_DATABASE_URI = 'postgres://zpsubtcgyhmhue:1558b0987b0ec9fa461f2072c176d2712e0498d7541cc965a5f0bb007766e18e@ec2-3-226-231-4.compute-1.amazonaws.com:5432/d88km9ic1ahkbe'


app = Flask("SmartBox Companion App",instance_relative_config=True)
app = Flask(__name__)

# ================== ON START DATABASE CONNECTION ======================
POSTGRES_URL = "localhost:5432"
POSTGRES_USER = "postgres"
POSTGRES_PW = "smartbox"
POSTGRES_DB = "smartbox"

DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(
    user=POSTGRES_USER,pw=POSTGRES_PW,url=POSTGRES_URL,db=POSTGRES_DB)

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
# silence the deprecation warning 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)
#SQLALCHEMY_DATABASE_URI = "postgresql://{}:{}@localhost:{}/{}".format(POSTGRES_USER, POSTGRES_PW, POSTGRES_URL, POSTGRES_DB)
# SQLALCHEMY_DATABASE_URI = 'postgres://zpsubtcgyhmhue:1558b0987b0ec9fa461f2072c176d2712e0498d7541cc965a5f0bb007766e18e@ec2-3-226-231-4.compute-1.amazonaws.com:5432/d88km9ic1ahkbe'
# app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# os.environ['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
# db = SQLAlchemy(app)
# if app.config is None:
#     app.config.from_object(BaseConfig)
# else:
#     app.config.from_object(BaseConfig)
# db.init_app(app)
# migrate = Migrate(app, db)

# ===================== HOMEPAGE REDIRECT =======================
@app.route('/')
def hello_world():
    # TODO: simply reroute this to /visualizer
    return redirect('visualizer')


# =============== DATA ENTRY AND RETRIEVAL REST API ====================

@app.route('/data_entry', methods=['POST'])
def enter_data():
    req_data = request.get_json()
    print("Got request")
    box = req_data['box_name']
    print(box)
    channels = req_data['channel_names']
    #print(channels)
    data = req_data['datapoints']
    #print(data)

    pictures = {}
    if 'pictures' in req_data:
        pictures = req_data['pictures']

    for timestamp, values in data.items():
        ts = datetime.utcfromtimestamp(int(float(timestamp))).strftime('%Y-%m-%d %H:%M:%S')
        value_floats = list(map(float, values.split(",")))
        for i in range(len(channels)):
            addData(box, channels[i], ts, value_floats[i])

    if pictures:
        print("Adding Pictures")
        for timestamp, pic64 in pictures.items():
            ts = datetime.utcfromtimestamp(int(float(timestamp))).strftime('%Y-%m-%d %H:%M:%S')
            addPicture(box, ts, pic64)

    db.session.commit()
    #displayTable(Data)
    print("Data Entered")

    return "Data Entered"

# ONLY FOR TESTING
@app.route('/picture_entry', methods=['POST'])
def enter_picture():
    req_data = request.get_json()
    print("Got Request")
    box = req_data['box_name']
    timestamp = req_data['time']
    ts = datetime.utcfromtimestamp(int(float(timestamp))).strftime('%Y-%m-%d %H:%M:%S')
    print(box)
    print(type(ts))
    pic64 = req_data['picture']


    print("Pic len", len(pic64))
    
    
    addPicture(box, ts, pic64)
    
    
    db.session.commit()
    #displayTable(Picture)
    print("Picture Entered")
    return "Picture Entered"



@app.route('/check_box', methods=['POST'])
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


@app.route("/get_data", methods=['GET'])
def get_data_request():
    box = request.args.get('box_name')
    start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
    
    result = get_data(box, start_time, end_time)
    result = [(row[0], row[1], row[2].strftime('%Y-%m-%d %H:%M:%S'), row[3], row[4])
                for row in result]
    

    return json.dumps(result)


# all_frames parameter should only be specified if needed.
@app.route("/get_pictures", methods=['GET'])
def get_pictures_request():
    print("Get pictures request")
    box = request.args.get('box_name')
    start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
    
    
    all_frames = request.args.get('all_frames')
    if all_frames == "True":
        all_frames = True
    else:
        all_frames = False


    result = get_pictures(box, start_time, end_time, all_frames)
    #print(result)

    result = [(row[0],row[1].strftime('%Y-%m-%d %H:%M:%S'), row[2])
                for row in result]
    
    return json.dumps(result)

@app.route("/get_box_channels", methods=['GET'])
def get_box_channels_request():
    box = request.args.get('box_name')
    
    result = get_box_channels(box)
    print(result)

    return json.dumps(result)

@app.route("/get_box_names", methods=['GET'])
def get_box_names_request():
    result = get_box_names()

    print(result)

    # TODO: return this in a post request
    return json.dumps(result)

# =========== WRAPPER FUNCTIONS FOR DATABASE QUERIES ===================

def addData(box, channel, time, value, label=None):
    data = Data(box_name=box, channel_name=channel, time=time, value=value, label=label)
    db.session.add(data)

def addPicture(box, ts, picture64):
    print(type(ts))
    pic = Picture(box_name=box, time=ts, picture=base64.b64decode(picture64))
    db.session.add(pic)

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

# start_time and end_time are in python datetime types
# returns a list of tuples (box_name, datetime (python format), base64 string picture)
# note: you must parse the datetime object before printing!
# all_frames parameter is if you want all pics between those times
# WARNING: with all=True, this will take a very long time.
def get_pictures(box, start_time, end_time, all_frames=False):
    print("getting pictures", box, start_time, end_time)
    query = "SELECT * FROM Picture p WHERE p.box_name = '{}' and p.time BETWEEN '{}' AND '{}';".format(box, start_time, end_time)
    result = list(db.engine.execute(text(query)))
    #print(result)
    
    # keep approximately 100 frames in the final gif. 
    # this helps the gif from not being massive!
    if not all_frames:
        mod = max(100, len(result)//100)
    else:
        mod = len(result)

    result = [(row[0],row[1], base64.b64encode(row[2]).decode('utf-8'))
                for i, row in enumerate(result) if i % mod == 0]
    print("returning  # frames = ", len(result))
    return result


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


class Picture(db.Model):
    box_name = db.Column(db.String(255), db.ForeignKey('box.box_name'),
                        primary_key=True)
    time = db.Column(db.DateTime, primary_key=True)
    picture = db.Column(db.LargeBinary)

    def __repr__(self):
        return '<Picture: {}, {}, {}>'.format(self.box_name, 
            self.time, len(self.picture))

# ========================= DATA VISUALIZATION DASH APP =============================
visualizer = dash.Dash(name="visualizer", server=app, 
                        url_base_pathname='/visualizer/',
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
        dbc.NavItem(dbc.NavLink("Data Visualizer", href="/visualizer/", external_link=True)),
        dbc.NavItem(dbc.NavLink("Labeling", href="/labeling/", external_link=True)),
        dbc.NavItem(dbc.NavLink("AL Configuration", href="/config/", external_link=True)),
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
    brand='SmartDash',
    brand_href='/visualizer/',
    sticky='top',
)
all_options = {
                    'occupancies': ['0','1','2','3','4','5','6',],
                    'activities': ['None','Running', 'Eating',]
                }

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
                    'height': '120vh'
                }
            )
        ],
        md=8),
        dbc.Col([
            dbc.Row([
                html.H3("Selected Data"),
                html.Pre(id='selected-data', style=styles['pre'], children="Please Select Data by dragging a box on the graph!"),  
                

                html.Div([
                    dcc.Dropdown(
                        id='oa_option',
                        options=[{'label': k, 'value': k} for k in all_options.keys()],
                        value='occupancies'
                    ),
                    html.Hr(),
                    dcc.Dropdown(id='oa_according_option'),
                    html.Hr(id='start_end_receive'),
                ]),

                dbc.Row(children=[], 
                    id="label-card-container-1"),

                dcc.Input(id="label", type="text", placeholder='Activity Label'),
                html.Br(),
                dbc.Button("Submit Label", id="label-submit", color="info", className="mr-1"),
            ],
            style=styles['40vh']),
            dbc.Row([
                html.H3("Video Feed from Selected Data"),
                html.Div([], id='selected-gif-container')
            ]),

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
                        for box in get_box_names()],
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

visualizer_body = dbc.Container(
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


visualizer.layout = html.Div([navbar, visualizer_body])
#visualizer.layout = html.Div([graph_select_form])

# ========================== VISUALIZER CALLBACKS ======================
@visualizer.callback([Output('channel_dropdown', 'options')],
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

# box, start_time, end_time as strings
# channels is either a list of channels or has ['All Channels']
def graph_data(box, start_time, end_time, channels):
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

    return fig

@visualizer.callback([Output('subplots-graph', 'figure')],
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
    
    fig = graph_data(box, start_time, end_time, channels)
    return [fig]


# Display the selected data on the right when you select on graph
@visualizer.callback(
    [Output('selected-data', 'children'),
     Output('selected-gif-container', 'children')],
    [Input('subplots-graph', 'selectedData')],
    [State("box_dropdown", "value")])
def display_selected_data(selectedData, box):
    if not selectedData:
        raise PreventUpdate
    # print("selectedData------------------------------>",selectedData)
    # print("type(selectedData)-------------------------------->",type(selectedData))
    rnge = selectedData['range']

    # weird string processing step to work on any subplot
    xx = [key for key in rnge.keys() if 'x' in key]
    print("found ", xx)


    start_time, end_time = rnge[xx[0]][0], rnge[xx[0]][1]
    
    gif_name = create_gif(box,start_time,end_time)

    gif_player = Gif.GifPlayer(
                        gif=gif_name,
                        still=gif_name[:-3]+".png"
                    )


    return [json.dumps(selectedData, indent=2),
            gif_player]


@visualizer.callback(
    Output('oa_according_option', 'options'),
    [Input('oa_option', 'value')])
def set_data_type(selected_option):
    return [{'label': i, 'value': i} for i in all_options[selected_option]]


@visualizer.callback(
    Output('oa_according_option', 'value'),
    [Input('oa_according_option', 'options')])
def set_data_type_accordingly(available_options):
    return available_options[0]['value']

 

@visualizer.callback(
    Output("label-card-container-1", "children"),
    [Input("label-submit", 'n_clicks'),
    Input('subplots-graph', 'selectedData')],
    [State("oa_according_option", 'value'),
    State("box_dropdown", "value")])
def populate_label(label_submit,selectedData,oa_according_val,box):
    # button  becuase somebody clicked the refresh button
    if not selectedData:
        raise PreventUpdate
    # print("selectedData",selectedData)
    # print("type(selectedData)",type(selectedData))
    rnge = selectedData['range']

    # weird string processing step to work on any subplot
    xx = [key for key in rnge.keys() if 'x' in key]
    print("found ", xx)


    start_time, end_time = rnge[xx[0]][0], rnge[xx[0]][1]

    if oa_according_val == 'None' or oa_according_val == '0':
        query = "UPDATE Data set label = '{}' where time between '{}' and '{}' and box_name='{}';".format(label_submit,start_time,end_time,box)
        result = db.engine.execute(text(query))
        # return [row for row in result]
    else:
        query = "UPDATE Data set label = '{}' where time between '{}' and '{}' and box_name='{}';".format(oa_according_val,start_time,end_time,box)
        result = db.engine.execute(text(query))
        # return [row for row in result]





# =========================== LABELING HELPERS =========================
def get_labels():
    return ["No People", "1 Person", "2 People", "3+ People"]

# =========================== LABELING WEB COMPONENT=====================

labeler = dash.Dash(name="labeling", server=app, 
                        url_base_pathname='/labeling/',
                        external_stylesheets=[dbc.themes.BOOTSTRAP])



labeler_body = dbc.Container([
    dbc.Row(children=[
        dbc.Button(
            "Run Active Learning", color="success", className="mr-1", id="run-AL-button"
        ),
        html.H1(
            id="AL_RUN"
        )
    ]),
    dbc.Row(children=[], 
        id="label-card-container",
        style={
            'width':'75vw',
    }),
    dbc.FormGroup([
        dbc.Label("Your Label", html_for="label_dropdown"),
        dcc.Dropdown(
            id="label_dropdown",
            options = [
                {"label": label, "value": label} 
                    for label in get_labels()
            ])
    ]),
    dbc.Row([
        dbc.Button("Submit", color="success", className="mr-1", id="submit-label-button"),
    ])
])

labeler.layout = html.Div([navbar, labeler_body])

# ======================== HELPER FUNCTIONS ========================

def create_gif(box, start_time, end_time):
    pics = get_pictures(box, start_time, end_time)
    frames = []

    for i in range(len(pics)):
        data = pics[i][2]
        data = base64.b64decode(data)
        frames.append(Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))))

    black_frame = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    outputGIF = io.BytesIO()
    full_gif = [black_frame] * 5 + frames
    gif_name = "./assets/gifs/{}.gif".format(time.time())
    frames[0].save(gif_name[:-4] + ".png", format="PNG")
    with open(gif_name, "w+b") as g:
        black_frame.save(g, format='GIF', append_images=full_gif, save_all=True, duration=100, loop=0)

    return gif_name


def get_next_label_card():
    query = get_next_AL_query()
    
    # no more!
    if not query:
        return None
    
    box, start_time, end_time = query

    # Time segment as title, (Data Graph), Gif all displayed in a card.
    heading = start_time + " - " + end_time
    fig = graph_data(box, start_time, end_time, ["All Channels"])
    gif_name = create_gif(box, start_time, end_time)

    card_content = [
        dbc.CardBody(
            [
                html.H5("Please Provide a label for the following segment", className="card-title"),
                html.H6(heading),
                html.Div([
                    Gif.GifPlayer(
                        gif=gif_name,
                        still=gif_name[:-3]+".png"
                    )
                ]),
                dcc.Graph(
                    figure=fig
                )
            ]
        ),
        
    ]

    return card_content

@labeler.callback([Output("label-card-container", "children")],
    [Input("submit-label-button", 'n_clicks')],
    [State("label_dropdown", "value")])
def populate_label_card(n, label):
    # button  becuase somebody clicked the refresh button
    if n and label:
        segment_label = label
        print(segment_label)

    card = get_next_label_card()
    if not card:
        return [html.H3("There are no more segments to label at this time. Thank you.")]
    return [card]




# return a segment:
# (box, start_time, end_time)
def get_next_AL_query():
    # TODO: clear out the gif folder.
    files = glob.glob('/assets/gifs/*')
    for f in files:
        os.remove(f)
    return ('Box0', '2020-03-22 21:11:00', '2020-03-22 21:15:00')





@labeler.callback([Output("AL_RUN", "children")],
    [Input("run-AL-button", 'n_clicks')])
def run_AL(n):
    print("Button clicked")
    # get all the data since the last time AL was run
    df = get_recent_data()
    print("Done retreiving data")
    print(df.head)

    # run DAE CPD to get the segments
    time_segments = run_dae_cpd(df)

    # pass each segment through a feature selector
    # features is now a n x 13743 matrix, with each row
    # representing a single segment
    features = get_features(time_segments)

    # get the AL classifier trained on the old dataset
    learner = get_AL_learner()

    print("Predicting Stuff!")
    predictions = learner.predict(np.nan_to_num(features.astype("float32")))

    print(predictions)






    return ("Done Updating", )
    


# =======================ACTIVE LEARNING METHODS========================

def get_recent_data():
    with open("assets/state.json") as f:
        last_pulled_json = json.load(f)
        last_pulled_date = last_pulled_json['last_pulled_date']

    print("Last pulled date: ", last_pulled_date)
    # Get current time
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("Now: ", now)

    df = get_df_data("Box0", last_pulled_date, now)
    
    print(df.head())

    # update json file with current date
    #last_pulled_json['last_pulled_date'] = now
    #with open("/assets/state.json", "w") as jsonFile:
    #    json.dump(last_pulled_json, jsonFile)

    return df

def get_df_data(box, start_time, end_time):
    # pull data from database between the last pulled date and now
    data = get_data(box, start_time, end_time)

    df = pd.DataFrame(data, columns=['box_name', 'channel_name', 'time', 'value', 'label'])
    #print("\ngot " + str(len(df)) + " rows of data. processing data......")
    df.sort_values(by='time', inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    return df

def run_dae_cpd(df):
    data = data_processing.load_dataset(df)
    values = data.values
    X, y = values[:, :-1], values[:, -1]    # X : Samples, Y: Labels

    ## COMMENTED THIS OUT TO PREVENT A 20 MIN DELAY
    #model = dae_cpd.dae(X, y).fit()
    #result = model.fit_predict()  # change point indexes
    result = [57, 211, 306, 359, 545, 691, 1176, 1319, 1412, 
                1820, 2277, 2470, 2559, 2696, 2872, 3179, 
                3270, 3357, 3564, 3645, 3772, 3976, 4148,
                4287, 4349, 4408, 4472, 4557, 4758, 4856, 
                4933]
    print(result)

    times = sorted(df['time'].unique())
    print(len(times))
    #print(times)

    segments = [] # list of timestamp segments
    segments.append((times[0], times[result[0]]))

    for i in range(1, len(result)):
         segments.append((times[result[i-1]], times[result[i]]))

    print(segments[:3])
    return segments

def get_features(time_segments):
    segment_features = np.empty((0, 13734), float)
    for start_time, end_time in time_segments:
        segment_data = get_df_data("Box0", start_time, end_time)
        features = tsfresh.extract_features(
                    segment_data, 
                    column_id="box_name",
                    column_sort="time", 
                    column_kind="channel_name", 
                    column_value="value")
        print(features)
        print(len(features))
        print(type(features))
        print(features.to_numpy().shape)
        segment_features = np.append(segment_features, 
                            features.to_numpy(), axis=0)
        print(segment_features)

    print("Done computing the features: ", segment_features.shape)
    return segment_features

# this learner will be trained on the old dataset
# apologies for the long AF function
# there is a ton of data cleaning here
# that is required since we are using 
# an old csv for training
# which is in a different labeling format
# please see the jupyter notebook
# to see more detailed explanation

def get_AL_learner():
    training_features_path = "./2020-05-30p1-segment-features.npy"
    if os.path.exists(training_features_path):
        segment_features = np.load(training_features_path)
    else:
        print("Getting AL Learner")
        # 1) Load the old dataset
        df = pd.read_csv('./2020-05-30p1.csv')
        df.sort_values(by='time', inplace=True)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        # 2) get the segments by seeing where the labels change

        start_indices = [df.iloc[0]['time']]

        for i in range(1, len(df)):
            if df.iloc[i]['label'] != df.iloc[i-1]['label']:
                start_indices.append(df.iloc[i]['time'])

        start_indices.append(df.iloc[-1]['time'])

        print(start_indices)

        segments = []
        for i in range(1, len(start_indices)):
            segments.append((start_indices[i-1], start_indices[i]))
            
        print(segments)

        # 3) Relabel and normalize all labels to numerals (0, 1, 2, 3, 4, 5)
        for segment in segments:
            print(df.loc[df['time'] == segment[0], 'label'].iloc[0])
            numPeopleinSegment = len(str(df.loc[df['time'] == segment[0], 'label'].iloc[0]).split(","))
            print(numPeopleinSegment)
            if str(df.loc[df['time'] == segment[0], 'label'].iloc[0]) == "0":
                numPeopleinSegment = 0
            df.loc[((segment[0] <= df['time']) & (df['time'] < segment[1])), 'label'] = numPeopleinSegment

        ### fix for last time stamp, annoying and hard coded. change this if training dataset changes.
        df.loc[df['time'] == start_indices[-1], 'label'] = 1

        print(df['label'].unique())


        # 4) get features for each segment
        ## this is very very similar to the get_features() 
        ## method, but now we want to pull data from 
        ## the old csv, not from the database
        ## so we need to rewrite it

        segment_features = np.empty((0, 13734), float)

        for index, segment in enumerate(segments):
            if index == len(segments) - 1:
                # only last segment fix, must include the last index
                # note the change in <= from < in second line
                segment_df = df.loc[((segment[0] <= df['time']) & 
                                    (df['time'] <= segment[1]))]
            else:
                segment_df = df.loc[((segment[0] <= df['time']) & 
                                    (df['time'] < segment[1]))]
                
            features = tsfresh.extract_features(
                            segment_df, 
                            column_id="box_name",
                            column_sort="time", 
                            column_kind="channel_name", 
                            column_value="value")
            print(features)
            print(len(features))
            print(type(features))
            print(features.to_numpy().shape)
            segment_features = np.append(segment_features, 
                                features.to_numpy(), axis=0)
            print(segment_features)
            
        print("Done computing the features: ", segment_features.shape)
        np.save(training_features_path, segment_features)


    # 5) train classifier on these features

    ## use the old csv file as training data for the
    ## random forest classifier
    ## then use the modAL library to conver it into
    ## an active learner

    labels = [5, 0, 4, 3, 1]
    X_training, y_training = np.nan_to_num(segment_features.astype('float32')), labels

    rf = RandomForestClassifier(random_state=1)

    #initialize learner
    learner = ActiveLearner(estimator= rf,  ## nn works awful, rf is best
                        query_strategy = margin_sampling,  ## margin sampling worked best
                        X_training=X_training, y_training=y_training)


    return learner














        


        

