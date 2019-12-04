import json

config = {
    "sensor_names": ['Timestamp', 'Photo1', 'Photo2', 'JoystickX', 'JoystickY', 
           'Sound', 'Temp'],
    "box_id": 0,
}

with open("config.json", 'w') as fp:
    json.dump(config, fp)