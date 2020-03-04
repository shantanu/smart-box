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
import cv2
import base64


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




print("ready to collect")
with open("config.json", 'r') as fp:
    config = json.load(fp)

sensor_names = config['channel_names']
channel_names = [x[0] for x in config['channel_names']]
box_name = config['box_name']

# MAKE CAMERA PORT SOMETHING NEGATIVE IF NOT IN USE
camera_port = int(config['camera_port'])

if camera_port >= 0:
    USING_CAMERA = True
else:
    USING_CAMERA = False

if USING_CAMERA:
    camera = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
    # Check if the webcam is opened correctly
    if not camera.isOpened():
        raise IOError("Cannot open webcam")


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

def get_picture_base64():
    return_value = False
    while not return_value:
        return_value, image = camera.read()
    
    # length 1,228,800 base 64 string to be sent in post request
    b64im = base64.b64encode(image)
    image_encoded = b64im.decode('utf-8')

    print("created base64 image len", len(image_encoded))
    return image_encoded



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


for _ in range(2):
    # run script for 10 seconds
    endtime = time.time() + 10
    
    post = {}
    post["box_name"] = box_name
    post["channel_names"] = channel_names

    datapoints = {}

    pictures = {}

    
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
        
        if USING_CAMERA:
            pictures[t] = get_picture_base64()

        while time.time() < t + sampling_period:
            time.sleep(0.05)

    post["datapoints"] = datapoints
    # if we took pictures
    if pictures:
        post['pictures'] = pictures
    
    

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

        
