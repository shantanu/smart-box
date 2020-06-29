## importing modules

import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling



## Load and preprocess data for shape(n_samples, n_features)

def load_dataset(df):
    df1 = df.drop(['box_name'], 1) ## drop these two columns

    df2 = df1.set_index('time')

    df3 = df2.groupby('channel_name')   ## group each variable together
    

    mag_x = df3.get_group('Magnet X')
    mag_y = df3.get_group('Magnet Y')
    mag_z = df3.get_group('Magnet Z')
    acc_x = df3.get_group('Accel X')
    acc_y = df3.get_group('Accel Y')
    acc_z = df3.get_group('Accel Z')
    humid= df3.get_group('Humidity')
    altid = df3.get_group('Approx. Altitude')
    press = df3.get_group('Pressure')
    temp = df3.get_group('Temperature')
    C = df3.get_group('C')
    B = df3.get_group('B')
    G = df3.get_group('G')
    R = df3.get_group('R')
    lumos = df3.get_group('Lumosity')
    cl_temp = df3.get_group('Color Temp (K)')
    aud = df3.get_group('Audio')
    pir = df3.get_group('PIR')


    ## create a new dataframe df5 with each variable as a separate column

    df5 =pd.DataFrame(columns = ['Magnet_X', 'Magnet_Y', 'Magnet_Z', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Humidity', 'Approx_Altitude', 'Pressure', 'Temperature', 'C', 'B', 'G', 'R', 'Lumosity', 'Color_Temp_(K)', 'Audio', 'PIR', 'label'])
    df5['Magnet_X'] = mag_x['value']
    df5['Magnet_Y'] = mag_y['value']
    df5['Magnet_Z'] = mag_z['value']
    df5['Accel_X'] = acc_x['value']
    df5['Accel_Y'] = acc_y['value']
    df5['Accel_Z'] = acc_z['value']
    df5['Humidity'] = humid['value']
    df5['Approx_Altitude'] = altid['value']
    df5['Pressure'] = press['value']
    df5['Temperature'] = temp['value']
    df5['C'] = C['value']
    df5['B'] = B['value']
    df5['G'] = G['value']
    df5['R'] = R['value']
    df5['Lumosity'] = lumos['value']
    df5['Color_Temp_(K)'] = cl_temp['value']
    df5['Audio'] = aud['value']
    df5['PIR'] = pir['value']
    df5['label'] = mag_x['label']
    
    
    ## convert text labels as categories
    label_set = df5['label'].astype('category').cat.codes 
    #print(label_set.shape)
    
    df6 = df5.drop('label', axis = 1)

    #create new column in datafram for Occupancy category
    df6['Occupancy'] = label_set
   
    return df6

# nomalization
def pre_processing(data):
    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    
    return x_scaled

# standard deviation and mean for plotting
def plt_stat(performance):
    perform = np.asarray(performance)
    acc_err = np.std(perform, axis = 0)
    acc_mean = np.mean(perform, axis = 0)
    
    return acc_err, acc_mean


#plotting the result an dsae
def display(err, mean):

    plt.figure(figsize = (10, 8))
    x = np.arange(len(err))
    plt.errorbar(x, mean, yerr=err, label = 'Init label = 100', marker = 'o', color = 'tab:red', ecolor = 'tab:blue', alpha = 0.8,  capsize = 2)
    plt.xlabel ('# of Queries', fontsize = 14)
    plt.ylabel ('Accuracy',  fontsize = 14)
    plt.legend(loc = 'lower right',  fontsize = 14)
    plt.savefig ('./Activity classification performance - accuracy vs. number of queries - Init label 100.png')

    return


