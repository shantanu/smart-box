import serial
import time
import traceback
import numpy as np

import pymongo
from pymongo import MongoClient

sampling_period = 1
box_id = 0

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

headers = ['Timestamp', 'Photo1', 'Photo2', 'JoystickX', 'JoystickY', 
           'Sound', 'Temp']
db_headers = ['_id', 'Photo1', 'Photo2', 'JoystickX', 'JoystickY', 
           'Sound', 'Temp']


def get_values():
    ser.write(b'r')
    arduinoData = ser.readline().decode('ascii')
    return arduinoData.strip()


try:
    collection_name = str(time.time()) + "_Data"
    for _ in range(10):
        # run script for 20 seconds
        endtime = time.time() + 10
        
        post = {}
        post["headers"] = headers[1:]
        ts_list = []
        rec = np.zeros(shape=(0, 6), dtype=int)
        post['box_id'] = box_id
        print("empty post", post)
        
        # sample every few seconds
        while time.time() < endtime:
            # write to txt file
            t = time.time()
            # data is a comma separated string that has 6 ints in it
            data = get_values()
            
            # append the sensor values to rec, and time stamp to ts
            data_ints = np.array(list(map(int, data.split(",")))).reshape((1,6))
            rec = np.vstack((rec, data_ints))
            ts_list.append(t)

            #print(rec)
            
            #collection.insert_one(post)

            while time.time() < t + sampling_period:
                time.sleep(0.05)
        post['rec'] = rec.tolist()
        post['ts_list'] = ts_list
        post['_id'] = ts_list[0]
        print(post)
        collection = db[collection_name]
        collection.insert_one(post)
except Exception:
    ser.close()
    tb = traceback.format_exc()
else:
    tb = "No error"
finally:
    ser.close()
    print(tb)


