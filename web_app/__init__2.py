import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc 
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from flask import Flask

server = Flask("my app")

@server.route('/hello', methods=['GET'])
def hello():
    print("hi there")
    return "Hello there"

app2 = dash.Dash(name='place1', server=server, url_base_pathname='/test1/')

app2.layout = html.Div([
        html.H1('This is a test1'),
        html.Button(id="hi", children="hi"),
        html.P("Test", id="test")
    ])

@app2.callback([Output('test', 'children')], [Input("hi", "n_clicks")])
def testy(clicks):
    return [str(clicks)]

app3 = dash.Dash(name='place2', server=server, url_base_pathname='/test2/')

app3.layout = html.Div([
        html.H1('This is test2')
    ])

server.run(port=8070, host='127.0.0.1')