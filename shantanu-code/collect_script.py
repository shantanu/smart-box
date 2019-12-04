############################################################
# THIS FILE READS DATA FROM THE ARDUINO SERIAL MONITOR AND 
# UPLOADS IT ONE AT A TIME TO THE DATABASE
# THIS IS PROBABLY NOT THE FILE YOU'RE LOOKING FOR.
# USE UPLOAD_CHUNKS.PY INSTEAD.
###########################################################

import serial
import time
import json

import pymongo
from pymongo import MongoClient

sampling_period = .5

print("initializing board")

"""USE THIS FOR LAPTOP"""
ser = serial.Serial('COM5', baudrate=9600, timeout=1)

"""USE THIS FOR RASPBERRY PI"""
#ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)


#===================================================

# allow arduino board to initialize fully before 
# collecting any data points
time.sleep(3)
print("Initializing Database")

cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]



print("ready to collect")
with open("config.json", 'r') as fp:
    config = json.load(fp)

headers = ['Timestamp'] + config['sensor_names']
db_headers = ['_id'] + config['sensor_names']


def get_values():
    ser.write(b'r')
    arduinoData = ser.readline().decode('ascii')
    return arduinoData.strip()


try:
    # run script for 20 seconds
    endtime = time.time() + 20
    filename = str(time.time()) + "_Data.txt"
    collection_name = filename[:-4]
    collection = db[collection_name]
    
    with open(filename, 'a') as datafile:
        # write file headers
        datafile.write(','.join(headers) + '\n')
        # sample every few seconds
        while time.time() < endtime:
            # write to txt file
            t = time.time()
            data = get_values()
            output = "{}, {}".format(t, data)
            datafile.write(output + "\n")
            
            
            # write to database post
            data_ints = list(map(int, data.split(",")))
            post = dict(zip(db_headers, [t] + data_ints))
            
            print(post)
            
            collection.insert_one(post)

            while time.time() < t + sampling_period:
                time.sleep(0.05)
except Exception:
    ser.close()
    
finally:
    ser.close()

