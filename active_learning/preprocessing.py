"""
description: data pre-processing, feature extraction and several shallow models
author: Tong
date: 4/14/2020
"""

import pandas as pd
from sklearn import preprocessing


def load_data(path='2020-03-23p2.csv'):

    channel_names = ["PIR", "Audio", "Color Temp (K)",
                     "Lumosity", "R", "G", "B", "C", "Temperature", "Pressure",
                     "Approx. Altitude", "Humidity", "Accel X", "Accel Y", "Accel Z",
                     "Magnet X", "Magnet Y", "Magnet Z"]

    df = pd.read_csv(path)
    times = df['time'].unique()
    feature_list = []

    for i in range(len(channel_names)):
        values = []
        for j in range(len(times)):
            values.append(df.at[i+j*len(channel_names), 'value'])
        feature_list.append(values)

    feature_matrix = pd.DataFrame(feature_list).T
    feature_matrix.columns = channel_names
    # feature_matrix['timestamp'] = times

    return feature_matrix


def pre_processing(data):
    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data.values)
    x_scaled = pd.DataFrame(x_scaled)
    return x_scaled


samples = load_data()
samples_scaled = pre_processing(samples)


