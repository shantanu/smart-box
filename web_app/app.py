import os
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib
import shutil
from datetime import datetime

app = Flask(__name__)

import pymongo
from pymongo import MongoClient

import numpy as np
import matplotlib.pyplot as plt

cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]

@app.route('/', methods=['GET'])
def select_collection():
    collections = db.collection_names()
    collection_dates = [datetime.fromtimestamp(float(x[:-5])).strftime('%m/%d/%Y %H:%M:%S')
                        for x in collections]
    print(collections)
    return render_template('collections_page.html',
                           collections=list(zip(collections, collection_dates)))

@app.route('/graph', methods=['GET'])
def graph_collection():
    collection = db[request.args.get('col', "")]
    print(collection)
    
    cursor = collection.find({}).sort('_id', pymongo.ASCENDING)
    
    
    
    return render_template('inside_collection.html',
                           cursor=cursor)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=True)