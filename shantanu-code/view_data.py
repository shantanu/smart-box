import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


#style.use('fivethirtyeight')
fig, ax = plt.subplots(6, 1, sharex='col')
fig.tight_layout()

print("initializing board")

"""USE THIS FOR LAPTOP"""
ser = serial.Serial('COM5', baudrate=9600, timeout=1)

"""USE THIS FOR RASPBERRY PI"""
#ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)

#============================================================

# allow arduino board to initialize fully before 
# collecting any data points
time.sleep(3)
print("ready to collect")


total_data = np.array([[], [], [], [], [], []])


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
    total_data = np.hstack((total_data, np_data))
    
        
    for j in range(6):
        ax[j].clear()
        ax[j].set_title(headers[j+1])
        
        ax[j].plot(list(range(len(total_data[j]))), (total_data[j]).flatten())
        
    

try:
    ani = animation.FuncAnimation(fig, animate)
    plt.show()
except Exception:
    ser.close()
    
finally:
    ser.close()

