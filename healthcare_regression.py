"""
In this exercise, I will be using healthcare data from kaggle (https://www.kaggle.com/shivan118/healthcare-analytics) to
build a model that can predict whether a new patient is going to have a favorable visit to the healthcare facility or not.
All the data is anonymous, so the conditions are labeled Var1, Var2, etc. This makes the data practically not very usable,
but is nice for an exercise.

As it turns out this data is super sparse and there are very few patientsIDs th. I think I would need more time to
figure out how to prune and set up the data to be able to do anything useful with it. Also would probably be useful
to make a more complicated and thoughtful model than a bunch of dense layers.
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


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
first_camp['id'] = first_camp.apply(lambda x: '%s_%s' % (int(x['Patient_ID']), int(x['Health_Camp_ID'])),axis=1)
first_camp = first_camp.drop(['Health_Camp_ID', 'Donation', 'Patient_ID'], axis=1)

second_camp = second_camp.sort_values(by=['Patient_ID'])
second_camp['id'] = second_camp.apply(lambda x: '%s_%s' % (x['Patient_ID'], x['Health_Camp_ID']),axis=1)
second_camp = second_camp.drop(['Health_Camp_ID', 'Patient_ID'], axis=1)


# For now we are only going to use x entries, as I am running this locally and don't have a ton of computing power.
x = 1500

patient_info = patient_info.iloc[:x]
first_camp = first_camp.iloc[:x]
second_camp = second_camp.iloc[:x]

# We can now make the train test splits for our data
x_train, x_test, y_train, y_test = train_test_split(patient_info, first_camp, test_size=0.5, shuffle=True, random_state=0)

model = define_model((5,))

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)