import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


#style.use('fivethirtyeight')
fig, ax = plt.subplots(6, 1, sharex='col')

print("initializing board")
ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)
# allow arduino board to initialize fully before 
# collecting any data points
time.sleep(3)
print("ready to collect")


total_data = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)


# visualize just Photoresistor 1 vs datapoints



headers = ['Timestamp', 'Photo1', 'Photo2', 'JoystickX', 'JoystickY', 
           'Sound', 'Temp']


def get_values():
    ser.write(b'r')
    arduinoData = ser.readline().decode('ascii')
    return arduinoData.strip()


def animate(i):
    global total_data
    t = time.time()
    data = get_values()
    np_data = np.array(list(map(int, data.split(',')))).reshape(6,1)
    #print(np_data)
    total_data = np.hstack((total_data, np_data))
    #print(total_data.shape)
    
    #photo_values.append(int(data.split(',')[0]))
    #print(photo_values)
        
    for j in range(6):
        #print(total_data[j][1:], range(len(total_data[j])-1))
        ax[j].clear()
        ax[j].plot(list(range(len(total_data[j])-1)), (total_data[j][1:]).flatten())
        
    

try:
    ani = animation.FuncAnimation(fig, animate)
    plt.show()
except Exception:
    ser.close()
    
finally:
    ser.close()

