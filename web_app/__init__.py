import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


import pymongo
from pymongo import MongoClient

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import json
from textwrap import dedent as d



cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
    },
    '40vh': {
        'height': '40vh', 
        'display': 'block'
    },
    'disappear': {
        'display': 'none',
    },


}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ====================== UTILITY FUNCTIONS ========================================
def collection_name_to_date(name):
    return datetime.fromtimestamp(float(name[:-5])).strftime('%m/%d/%Y %H:%M:%S')

# ========================= LAYOUT ==========================================

# represents the URL bar, doesn't render anything
location = dcc.Location(id='url', refresh=False)

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

# pills nav on left side
nav = dbc.Nav(
    children=[],
    pills=True,
    vertical=True,
    id="pills",

)

dynamic_graphs_layout = [
    dbc.Row([
        html.H1("Please pick a collection from left.", id="collection-title"),
        dbc.Button("Pause Live Database", id="live-button", style=styles['disappear'], disabled=True),
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

body = dbc.Container(
    [
        dbc.Row([
            # nav with pills
            dbc.Col(   
                children=[
                    nav,
                ],
                md=2,
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


app.layout = html.Div([location, hidden_vars, navbar, body])


# =====================  CALLBACKS ================================


# Also Update graph when new collection is selected (pathname changes)
@app.callback([Output('collection-title', 'children'),
                Output('subplots-graph', 'figure'),
                Output('prev-url', 'children')],
              [Input('url', 'pathname'),
              Input('interval-component', 'n_intervals')],
              [State('prev-url', 'children'),
              State('paused', 'children'),
              State('live', 'children')])
def update_graphs(pathname, n, prev_url, paused, live):
    # on startup
    if not pathname:
        raise PreventUpdate
    if not n:
        raise PreventUpdate
    
    # check if url has changed: then always refresh
    if pathname != prev_url:
        print("\tprev_url is", prev_url)
        print("\tpathname is", pathname)
        print("\tURL fired graph refresh")
        return display_graphs(pathname) + (pathname,)
    
    # if here, then interval fired.
    # only update if graph is not paused
    print("\tInterval Fired graph refresh")
    if (paused == "True") or (live == "False"):
        print("\tGraph updating is paused or not live - nothing is updated")
        print("paused: ", paused)
        print("live: ", live)
        raise PreventUpdate
    
    print("\tGraph is updated becuase live and not paused")
    return display_graphs(pathname) + (pathname,)
    
# Actually do the graphing
def display_graphs(pathname=None):
    # Update the title of the page
    if (pathname == None) or (pathname == "/"):
        collection_name = "Please Pick a Collection from the Left"
        return (collection_name, None)
    
    print("\tDISPLAYING GRAPHS")
    collection_name = pathname.split("/")[-1]

    # Create the subplots graph
    collection = db[collection_name]
    #print(collection)
    
    cursor = collection.find({ '_id': { '$gt': 0 } } ).sort('_id', pymongo.ASCENDING)
    ts = np.array([])
    rec = np.zeros(shape=(0,6))
    headers = []
    
    for post in cursor:
        ts = np.append(ts, post['ts_list'])
        rec = np.vstack((rec, post['rec']))
        headers = post['sensor_names']
    
    ts = [datetime.utcfromtimestamp(t) for t in ts]
    rec = rec.T
    

    #print(rec)

    fig = make_subplots(rows=len(headers), cols=1, shared_xaxes=True,
                subplot_titles=headers)

    for i in range(len(headers)):
        fig.append_trace(go.Scatter(
            x=ts,
            y=rec[i].tolist(),
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
    
    print("\tGRAPHS RETURNED")
    return (collection_name_to_date(collection_name), fig)

# Check pills for updates every 15 seconds through the interval component
@app.callback(Output('pills', 'children'),
              [Input('interval-component', 'n_intervals')])
def load_collection_names(n):
    #if n == None:
    #    raise PreventUpdate

    list_of_names = db.list_collection_names()

    # display the most recent at the top of the list
    list_of_names.sort(reverse=True)


    print("Pills Refresh Callback called!")
    return [dbc.NavItem(
                dbc.NavLink(collection_name_to_date(x), href=(''.join(["/",x]))),
                id=''.join(["pills_", x]),
            ) 
            for x in list_of_names]

# Display the selected data on the right when you select on graph
@app.callback(
    Output('selected-data', 'children'),
    [Input('subplots-graph', 'selectedData')])
def display_selected_data(selectedData):
    if not selectedData:
        raise PreventUpdate
    return json.dumps(selectedData, indent=2)

# Update the Live Status (and Button)
@app.callback([Output('live-button', 'children'),
                Output('live-button', 'style'),
                Output('live-button', 'color'),
                Output('live', 'children')],
              [Input('interval-component', 'n_intervals'), 
              Input('url', 'pathname')],
              )
def check_if_live (n, url):
    if (not url) or (not n):
        raise PreventUpdate
    collection_name = url.split("/")[-1]

    if not collection_name:
        raise PreventUpdate
    # Create the subplots graph
    collection = db[collection_name]
    
    cursor = collection.find_one({ '_id': { '$eq': -1 } } )
    print("live cursor", cursor)
    live = False
    if cursor:
        live = cursor['live']


    if live:
        return ("Live", {'display': 'block'}, 'danger', "True")
        #return ("True",)
    else:
        return ("", {'display': 'none'}, '', "False")
        #return ("False",)

@app.callback([Output('label-submit-count', 'children'),
                Output('label', 'value')],
                [Input('label-submit', 'n_clicks'),
                Input('url', 'pathname')],
                [State('label', 'value'),
                State('selected-data', 'children')])

def submit_label (n, url,  label, selected_data):
    if (not url) or (not n):
        raise PreventUpdate
    if not label:
        raise PreventUpdate

    collection_name = url.split("/")[-1]

    if not collection_name:
        raise PreventUpdate
    collection = db[collection_name]

    

    selected_data = json.loads(selected_data)
    post_id = collection.find_one({"_id": {"$exists": True}}, sort=[("_id", 1)])["_id"] - 1
    post = {}
    post["_id"] = post_id
    post["left"] = selected_data['range']['x'][0]
    post['right'] = selected_data['range']['x'][1]
    post['label'] = label

    collection.insert_one(post)

    return (str(int(n) + 1), "")

# Update the list of segments stored in the database
@app.callback([Output('database-labels', 'children')],
                [Input('interval-component', 'n_intervals'),
                 Input('label-submit-count', 'children'),
                 Input('url', 'pathname')],
                 )

def update_database_labels(n, k, url):
    if (not url) or (not n):
        raise PreventUpdate
    collection_name = url.split("/")[-1]

    if not collection_name:
        raise PreventUpdate
    collection = db[collection_name]

    cursor = collection.find({ '_id': { '$lt': -1 } } )
    if cursor == None:
        raise PreventUpdate

    database_labels = []
    for post in cursor:
        database_labels.append(
            dbc.ListGroupItem(
                str(post['left']) + " - " + str(post['right']) + " : " + post['label']
            )
        )
                
    
    return (database_labels,)


"""
# Hover and Click Data Functions
# draw vertical line on all subplots instead of just the first one
@app.callback(
    Output('subplots-graph', 'figure'),
    [Input('subplots-graph', 'hoverData')],
    [State('subplots-graph', 'figure')])
def display_hover_data(hoverData, fig):
    if hoverData == None:
        raise PreventUpdate
    
    # see this link to do this:
    # https://community.plot.ly/t/vertical-line-on-hover-in-all-subplots/23685/3

"""


    

"""
@app.callback(
    Output('click-data', 'children'),
    [Input('subplots-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)
"""


# ========================= NECESSARY FUNCTIONS =========================

if __name__ == '__main__':
    app.run_server(debug=True)


# REFERENCES
#https://community.plot.ly/t/show-and-tell-dash-bootstrap-components/16614/15
#https://dash-bootstrap-components.opensource.faculty.ai/
