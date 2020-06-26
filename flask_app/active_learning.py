import numpy as np 
import pandas as pd 
import sklearn as skl 
import json
from datetime import datetime

from app import get_data

def get_recent_data():
    with open("/assets/state.json") as f:
        last_pulled_json = json.load(f)
        last_pulled_date = last_pulled_json['last_pulled_date']

    print("Last pulled date: ", last_pulled_date)
    # Get current time
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("Now: ", now)

    # pull data from database between the last pulled date and now
    data = get_data("Box0", last_pulled_date, now)

    # update json file with current date
    #last_pulled_json['last_pulled_date'] = now
    #with open("/assets/state.json", "w") as jsonFile:
    #    json.dump(last_pulled_json, jsonFile)

    df = pd.DataFrame(data, columns=['box_name', 'channel_name', 'time', 'value', 'label'])
    #print("\ngot " + str(len(df)) + " rows of data. processing data......")
    df.sort_values(by='time', inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    print(df.head())

    return df

    

