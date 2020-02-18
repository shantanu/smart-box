############################################################
# THIS FILE READS DATA FROM THE ARDUINO SERIAL MONITOR AND 
# UPLOADS IT IN CHUNKS OF 10 SECS TO THE DATABASE
# CONNECTION TO DATABASE IS THROUGH A FLASK SERVER RUNNING ON LOCALHOST
###########################################################

import serial
import time
import traceback
import numpy as np
import json
import requests
import sys
from requests.exceptions import HTTPError


sampling_period = 1

print("initializing board")

"""USE THIS FOR LAPTOP"""
ser = serial.Serial('COM5', baudrate=115200, timeout=1)

"""USE THIS FOR RASPBERRY PI"""
#ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)


#===================================================

# allow arduino board to initialize fully before 
# collecting any data points
time.sleep(3)
#print("Initializing Database")

#cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
#db = cluster["test"]



print("ready to collect")
with open("config.json", 'r') as fp:
    config = json.load(fp)

sensor_names = config['channel_names']
channel_names = [x[0] for x in config['channel_names']]
box_name = config['box_name']


# change this when server is hosted.
url = "http://127.0.0.1:5000"
headers = {'content-type': 'application/json'}



def get_values():
    arduinoData = ""
    try:
        ser.write(b'r')
        arduinoData = ser.readline().decode('ascii')
        if not arduinoData:
            arduinoData = ser.readline().decode('ascii')
    except Exception as err:
        print(f'Serial error occurred: {err}')  # Python 3.6
        
    
    print("data: ", arduinoData)
    return arduinoData.strip()




"""
#CHECK IF THE DATABASE HAS THE BOXNAME AND ALL THE CHANNEL NAMES HERE
"""

post = {'box_name': box_name, 
        'channel_names': config['channel_names']}

try: 
    print("Sending request")
    r = requests.post(url+"/check_box", data=json.dumps(post), headers=headers)
    r.raise_for_status()

except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')  # Python 3.6
    ser.close()
except Exception as err:
    print(f'Other error occurred: {err}')  # Python 3.6
    ser.close()
else:
    print('Success!')
    print("BOX VALIDATED!")



    """""
    NOW THE FUN STARTS! LET'S COLLECT DATA!!
    """""


    # collect data for 100 seconds - 10 chunks of 10 seconds each.


for _ in range(30):
    # run script for 10 seconds
    endtime = time.time() + 10
    
    post = {}
    post["box_name"] = box_name
    post["channel_names"] = channel_names

    datapoints = {}

    
    #rec = np.zeros(shape=(0, len(channel_names)), dtype=int)
    
    # sample every few seconds
    while time.time() < endtime:
        # write to txt file
        t = np.rint(time.time(), casting="safe")
        # data is a comma separated string that has 6 ints in it
        data = get_values()
        
        # append the sensor values to rec, and time stamp to ts
        #data_floats = np.array(list(map(float, data.split(",")))).reshape((1,len(channel_names)))
        # do this on the server side!
        #datapoints[t] = data_floats
        datapoints[t] = data
        

        while time.time() < t + sampling_period:
            time.sleep(0.05)

    post["datapoints"] = datapoints
    
    

    #print(post)
    try:
        r = requests.post(url+"/data_entry", data=json.dumps(post), headers=headers)
        r.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        ser.close()
    else:
        print("Data Submitted successfully!")

    print("Done!")

        
