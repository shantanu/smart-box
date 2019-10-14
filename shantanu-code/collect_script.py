import serial
import time

sampling_period = .5

print("initializing board")
ser = serial.Serial('COM5', baudrate=9600, timeout=1)
# allow arduino board to initialize fully before 
# collecting any data points
time.sleep(3)
print("ready to collect")

headers = ['Timestamp', 'Photo1', 'Photo2', 'JoystickX', 'JoystickY', 
           'Sound', 'Temp']


def get_values():
    ser.write(b'r')
    arduinoData = ser.readline().decode('ascii')
    return arduinoData.strip()


try:
    # run script for 20 seconds
    endtime = time.time() + 20
    filename = str(time.time()) + "_Data.txt"
    with open(filename, 'a') as datafile:
        # write file headers
        datafile.write(','.join(headers) + '\n')
        # sample every few seconds
        while time.time() < endtime:
            t = time.time()
            data = get_values()
            output = "{}, {}".format(t, data)
            datafile.write(output + "\n")
            print(output)

            while time.time() < t + sampling_period:
                time.sleep(0.05)
except Exception:
    ser.close()
    
finally:
    ser.close()

