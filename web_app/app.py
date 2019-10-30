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
import seaborn as sns
import io
import base64

cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]

@app.route('/', methods=['GET'])
def select_collection():
    collections = db.collection_names()
    collection_dates = [datetime.fromtimestamp(float(x[:-5])).strftime('%m/%d/%Y %H:%M:%S')
                        for x in collections]
    pass_in = list(zip(collections, collection_dates))
    pass_in.sort(key=lambda x: x[1], reverse=True)
    print(pass_in)
    #print(collections)
    return render_template('collections_page.html',
                           collections=pass_in)

@app.route('/graph', methods=['GET'])
def graph_collection():
    collection_name = request.args.get('col', "")
    collection_string = datetime.fromtimestamp(
        float(collection_name[:-5])).strftime('%m/%d/%Y %H:%M:%S')
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
    
    rec = rec.T
    #headers = cursor[0]['headers'][1:]
    
    sns.set()
    fig, ax = plt.subplots(6, 1, sharex=True,
                           figsize=(8, 16), dpi=80)
    for c, axis in enumerate(ax):
        axis.set_title(headers[c])
        axis.plot(ts, rec[c], label=headers[c])
    
    fig.suptitle(collection_string + "\n\n")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    buffer = b''.join(buf)
    b2 = base64.b64encode(buffer)
    fig2 = b2.decode('utf-8')
    
    return render_template('inside_collection.html', cursor=cursor,
                           fig=fig2)
        
    
    
        
    
    
        
    
    
    
    return render_template('inside_collection.html',
                           cursor=cursor)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=True)