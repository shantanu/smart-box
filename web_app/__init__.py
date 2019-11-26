import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

import pymongo
from pymongo import MongoClient

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
        'overflowX': 'scroll'
    }
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ====================== PRE-LAYOUT ========================================
def collection_name_to_date(name):
    return datetime.fromtimestamp(float(name[:-5])).strftime('%m/%d/%Y %H:%M:%S')

def load_collection_names():
    list_of_names = db.list_collection_names()

    # display the most recent at the top of the list
    list_of_names.sort(reverse=True)


    print("Callback called!")
    return [dbc.NavItem(
                dbc.NavLink(collection_name_to_date(x), href=(''.join(["/",x]))),
                id=''.join(["pills_", x]),
            ) 
            for x in list_of_names]

# ========================= LAYOUT ==========================================
# represents the URL bar, doesn't render anything
location = dcc.Location(id='url', refresh=False)


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
    children=load_collection_names(),
    pills=True,
    vertical=True,
    id="pills",

)

dynamic_graphs_layout = [
    dbc.Row([
        html.H1("Please pick a collection from left.", id="collection-title")
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
                html.H3("Click On A Data Point"),
                html.Pre(id='click-data', style=styles['pre'])
            ],
            style={'height': '40vh', 'display': 'block'}),
            dbc.Row([
                html.H3("Hover Data"),
                html.Pre(id='hover-data', style=styles['pre'])
            ],
            style={'height': '40vh', 'display': 'block'})
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


app.layout = html.Div([location, navbar, body])


# =====================  CALLBACKS ================================
@app.callback([dash.dependencies.Output('collection-title', 'children'),
                dash.dependencies.Output('subplots-graph', 'figure')],
              [dash.dependencies.Input('url', 'pathname')])
def display_graphs(pathname):
    # Update the title of the page
    if (pathname == None) or (pathname == "/"):
        collection_name = "Please Pick a Collection from the Left"
        return (collection_name, None)
    
    collection_name = pathname.split("/")[-1]

    # Create the subplots graph
    collection = db[collection_name]
    #print(collection)
    
    cursor = collection.find({}).sort('_id', pymongo.ASCENDING)
    ts = np.array([])
    rec = np.zeros(shape=(0,6))
    headers = []
    
    for post in cursor:
        ts = np.append(ts, post['ts_list'])
        rec = np.vstack((rec, post['rec']))
        headers = post['headers']
    
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
    
    #print(collection_name_to_date(collection_name))

    return (collection_name_to_date(collection_name), fig)

@app.callback(
    Output('hover-data', 'children'),
    [Input('subplots-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('subplots-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



# ========================= NECESSARY FUNCTIONS =========================

if __name__ == '__main__':
    app.run_server(debug=True)


# REFERENCES
#https://community.plot.ly/t/show-and-tell-dash-bootstrap-components/16614/15
#https://dash-bootstrap-components.opensource.faculty.ai/
