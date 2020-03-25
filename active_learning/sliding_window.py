import pandas as pd
import numpy as np

# need to read in csv file
# output format: list of numpy arrays:
# each row in the array is a time stamp, 18 sensors wide.
# sensors are in this order
channel_names =  [ "PIR", "Audio", "Color Temp (K)",
    "Lumosity","R","G","B","C","Temperature","Pressure",
    "Approx. Altitude","Humidity","Accel X","Accel Y","Accel Z",
    "Magnet X","Magnet Y","Magnet Z"]


# you can change this file, but currently this 
df = pd.read_csv('2020-03-23p2.csv')

times = df['time'].unique()
times.sort()

slider = []
for i in range(4541, len(times)-19):
    # window
    window = np.empty((20, 18))
    for j in range(20):
        # sensors
        for k in range(18):
            print(i, j, k)
            window[j, k] = df[(df['time']==times[j+i]) & 
                              (df['channel_name']==channel_names[k])]['value']
    slider.append(window)
    
print(slider)