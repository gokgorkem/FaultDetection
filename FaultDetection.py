from google.colab import files
  
  
uploaded = files.upload()

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

import io
data = pd.read_csv(io.BytesIO(uploaded['test.csv']))
print(data)

timeline = data['timeline']

X = data.drop('timeline', axis=1)

X

y = timeline

sum(y)/10000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = Sequential()
model.add(Dense(3, activation='relu', input_dim=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, validation_data=(X_test, y_test), epochs=10, verbose=1)

!mkdir -p saved_model
model.save('saved_model/my_model')

workstation_id = 5
ariza_id = 5

a_test = pd.DataFrame({"workstation_id":[workstation_id],"ariza_id":[ariza_id]})

print(a_test)

print(a_test)

a_test  = scaler.transform(a_test)

print(a_test)

deneme = model.predict(a_test)
print(deneme)

print (int(deneme*100))

