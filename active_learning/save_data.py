import pandas as pd
import numpy as np
import requests

print("Please make sure the ssh tunnel is running between port 5000 and server port 5000 before resuming.\n")
start_time = input("\nPlease enter the start time YYYY-MM-DD HH:MM:SS -> ")
end_time = input("\nPlease enter the end time YYYY-MM-DD HH:MM:SS -> ")

output_file = input("\nPlease enter the name of the name of the csv file (don't include .csv)-> ")

url = "http://localhost:5000/get_data"
print("\ngetting data.....")
r = requests.get(url, params={'box_name': 'Box0', 'start_time': start_time, 'end_time': end_time})
r.json()

df = pd.DataFrame(r.json(), columns=['box_name', 'channel_name', 'time', 'value', 'label'])
print("\ngot " + str(len(df)) + " rows of data. processing data......")
df.sort_values(by='time', inplace=True)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
df.head()

print("\nsaving data.....")

df.to_csv(output_file + ".csv")

