import cv2
import sys
from matplotlib import pyplot as plt
import time
import base64
import numpy as np
import requests
from requests.exceptions import HTTPError
import json

camera_port = 1
#camera = cv2.VideoCapture(camera_port)
camera = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open webcam")
start = time.time()
return_value, image = camera.read()
end = time.time()

print("We take a picture of you, check the folder")
#cv2.imwrite("image.png", image)

camera.release() # Error is here
cv2.destroyAllWindows()

b64im = base64.b64encode(image)
image_encoded = b64im.decode('utf-8')

print("Took", end-start)
#print("base64", b64im)
print("len", len(image_encoded))
print("type", type(image_encoded))


#frameRGB = image[:,:,::-1] # BGR => RGB
#plt.imshow(frameRGB)
#plt.show()




# change this when server is hosted.
url = "http://127.0.0.1:5000"
headers = {'content-type': 'application/json'}
post = {'box_name': "Box0", 
        'time': np.rint(time.time(), casting="safe"),
        'picture': image_encoded}

try:
    r = requests.post(url+"/picture_entry", data=json.dumps(post), headers=headers)
    r.raise_for_status()
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')  # Python 3.6