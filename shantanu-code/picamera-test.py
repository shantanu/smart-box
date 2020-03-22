import time
from picamera import PiCamera
from datetime import datetime, timedelta
import base64
import numpy as np

def get_picture_base64(pic_as_array):
    # length 1,228,800 base 64 string to be sent in post request
    b64im = base64.b64encode(pic_as_array)
    image_encoded = b64im.decode('utf-8')

    print("created base64 image len", len(image_encoded))
    return image_encoded

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24

starttime = time.time()

# note that resolution here is reversed
output = np.empty((480, 640, 3), dtype=np.uint8)
imagesb64 = []
for _ in range(5):
    camera.capture(output, 'rgb')
    imagesb64.append(get_picture_base64(output))
    time.sleep(2 - ((time.time() - starttime) % 2.0))
print(time.time() - starttime)
print(len(imagesb64[0]))

    
    
    
    
    