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
import base64
import threading
import subprocess
from PIL import Image

sampling_period = 1

ARDUINO_PORT = '/dev/ttyACM0' # use 'COM5' with windows
SERVER_URL = 'http://39e5b6eb.ngrok.io'


#==================== SETUP BOARD, CONFIG, CAMERA =================
def setup(url):
    global ARDUINO_PORT
    global box_name
    global sensor_names
    global channel_names
    global USING_CAMERA
    global headers
    global ser
    
    print("initializing board")

    """USE THIS FOR LAPTOP"""
    #ser = serial.Serial('COM5', baudrate=115200, timeout=1)

    """USE THIS FOR RASPBERRY PI"""
    ser = serial.Serial(ARDUINO_PORT, baudrate=115200, timeout=1)


    #===================================================

    # allow arduino board to initialize fully before 
    # collecting any data points
    time.sleep(3)




    print("configuring settings from json file")
    with open("config.json", 'r') as fp:
        config = json.load(fp)
    
    sensor_names = config['channel_names']
    channel_names = [x[0] for x in config['channel_names']]
    box_name = config['box_name']
    
    
    # MAKE USING_CAMERA 0 if not in use.
    USING_CAMERA = bool(config['USING_CAMERA'])

    if USING_CAMERA:
        print("Using Camera - make sure its plugged in")

    
    headers = {'content-type': 'application/json'}
    
    print("server url: ", url)
    
    check_box()


def check_box():
    """
    #CHECK IF THE DATABASE HAS THE BOXNAME AND ALL THE CHANNEL NAMES HERE
    """

    post = {'box_name': box_name, 
            'channel_names': sensor_names}

    try: 
        print("Sending request")
        r = requests.post(SERVER_URL+"/check_box", data=json.dumps(post), headers=headers)
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




setup(SERVER_URL)








#===========================GET DATA FROM ARDUINO================
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

#========================TAKING AND ENCODING PICTURES ============================
def get_picture_base64(pic_as_array):
    # length 1,228,800 base 64 string to be sent in post request
    b64im = base64.b64encode(pic_as_array)
    image_encoded = b64im.decode('utf-8')

    print("created base64 image len", len(image_encoded))
    return image_encoded

# returns array with 10 base64 pictures
def take_10_pictures(imagesb64):
    
    for i in range(11):
        ret = subprocess.run(["fswebcam", "/home/pi/smart-box/shantanu-code/pics/pic{}.jpg".format(i),
                      "-q", "-r", "640x480", "--no-banner", "--skip", "2", "--set", "brightness=50%"])
    
    for i in range(11):
        image = np.asarray(Image.open("/home/pi/smart-box/shantanu-code/pics/pic{}.jpg".format(i)))
        imagesb64.append(get_picture_base64(image))
        
    return imagesb64

#==================SEND DATA TO SERVER=========================
def send_data(datapoints, picture_thread, picturesb64):
    post = {}
    post["box_name"] = box_name
    post["channel_names"] = channel_names
    post['datapoints'] = datapoints
    
    if USING_CAMERA:
        picture_thread.join()
        pictures = {}
        for i, ts in enumerate(datapoints.keys()):
            print(i, ts)
            pictures[ts] = picturesb64[i]
        
        post['pictures'] = pictures
        
    
    print(post.keys())
    
    try:
        r = requests.post(SERVER_URL+"/data_entry", data=json.dumps(post), headers=headers)
        r.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        ser.close()
    else:
        print("Data Submitted successfully!")
    
    print("Done!")
    
    
    

"""""
NOW THE FUN STARTS! LET'S COLLECT DATA!!
"""""


# collect data for 100 seconds - 10 chunks of 10 seconds each.


for _ in range(10):
    # run script for 10 seconds
    endtime = time.time() + 10
    
    if USING_CAMERA:
        imagesb64 = []
        pictures_thread = threading.Thread(target=take_10_pictures, args=(imagesb64,))
        pictures_thread.start()

    datapoints = {}
    
    #rec = np.zeros(shape=(0, len(channel_names)), dtype=int)
    
    # sample every few seconds
    while time.time() < endtime:
        # write to txt file
        t = np.rint(time.time(), casting="safe")
        # data is a comma separated string that has 6 ints in it
        data = get_values()
        
        # append the sensor values to rec, and time stamp to ts
        # data_floats = np.array(list(map(float, data.split(",")))).reshape((1,len(channel_names)))
        # do this on the server side!
        # datapoints[t] = data_floats
        datapoints[t] = data
        

        while time.time() < t + sampling_period:
            time.sleep(0.05)
            
    if USING_CAMERA:
        send_thread = threading.Thread(target=send_data, args=(datapoints, pictures_thread, imagesb64))
    else:
        send_thread = threading.Thread(target=send_data, args=(datapoints, None, None))
    send_thread.start()
    

