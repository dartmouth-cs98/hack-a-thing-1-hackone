'''
In this exercise, I will be using healthcare data from kaggle (https://www.kaggle.com/shivan118/healthcare-analytics) to
build a model that can predict whether a new patient is going to have a favorable visit to the healthcare facility or not.
All the data is anonymous, so the conditions are labeled Var1, Var2, etc. This makes the data practically not very usable,
but is nice for an excersize
'''

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# First we will load the health care data into pandas dataframes
patient_info = pd.read_csv('data/healthcare/Patient_Profiles.csv')
first_camp = pd.read_csv('data/healthcare/First_Health_Camp_Attended.csv')
second_camp = pd.read_csv('data/healthcare/Second_Health_Camp_Attended.csv')


# We can then sort the dataframes by patient id and dorp unnecessary columns
# Concatenation trick from https://stackoverflow.com/questions/39291499/how-to-concatenate-multiple-column-values-into-a-single-column-in-panda-datafram


patient_info = patient_info.sort_values(by=['Patient_ID'])
patient_info['id'] = patient_info.apply(lambda x: '%s_%s' % (x['Patient_ID'], x['Health_Camp_ID']),axis=1)
patient_info = patient_info.drop(['Registration_Date', 'Patient_ID', 'Health_Camp_ID'], axis=1)


first_camp = first_camp.sort_values(by=['Patient_ID'])
first_camp['id'] = first_camp.apply(lambda x: '%s_%s' % (x['Patient_ID'], x['Health_Camp_ID']),axis=1)

first_camp = first_camp.drop(['Health_Camp_ID', 'Donation', 'Patient_ID'], axis=1)

second_camp = second_camp.sort_values(by=['Patient_ID'])
second_camp['id'] = second_camp.apply(lambda x: '%s_%s' % (x['Patient_ID'], x['Health_Camp_ID']),axis=1)

second_camp = second_camp.drop(['Health_Camp_ID', 'Patient_ID'], axis=1)


def define_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(.001)
    model.compile(loss='mse',
                  optimizer=optimizer
                  )
    return model

model = define_model((11,))
print(model.summary())