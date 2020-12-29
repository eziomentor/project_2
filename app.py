#!/usr/bin/env python
# coding: utf-8

# In[71]:


from flask import Flask
from flask import request, jsonify
from keras.models import Sequential
from flask_restful import Api, Resource, reqparse
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import pandas as pd
import joblib
from flask_restful import Resource


# In[72]:


def model_f(X,Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(Adam(lr=0.001), 'categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=1)
    
    return model

if  __name__ == '__main__':
    
    inner = pd.read_csv('Faulty_inner.csv')
    outer = pd.read_csv('Faulty_outer.csv')
    healthy = pd.read_csv('healthy.csv')
    
    pd.DataFrame(healthy)
    pd.DataFrame(outer)
    pd.DataFrame(inner)
    
    merge = pd.merge(inner, outer, how="outer")
    dataset = pd.merge(merge, healthy, how="outer")
    
    X = dataset.Vibration
    x_norm = (X - np.min(X))/(np.max(X)-np.min(X))
    X = x_norm
    
    

    Y = pd.get_dummies(dataset.Condition)
    Y = Y.values
    
    mdl = model_f(X,Y)


# y_pred = mdl.predict(np.array([0.005]))
# y_pred

# In[73]:


mdl.save('my_model.h5')


# In[74]:


model = keras.models.load_model('my_model.h5')


# name = 0.025
# name = np.array([name])
# y_pred = model.predict(name)
# y_new = np.argmax(y_pred)
# y_pred
# y_new

# In[ ]:


from flask import Flask, request


app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
     parser = reqparse.RequestParser()
     parser.add_argument('data', type=float)
     data = parser.parse_args()
     
    
     x_new = np.fromiter(data.values(), dtype=float)
     
        
        
     y_pred = model.predict(x_new)
     y_new = np.argmax(y_pred)
        
     if y_new == 0:
        condition = 'Healthy'
        return 'The Condition is {}'.format(condition)
     
     elif y_new == 1:
          condition = 'Inner race Faulty'
          return 'The Condition is {}'.format(condition)
     
     elif y_new == 2:
          condition = 'Outer race Faulty'
          return 'The Condition is {}'.format(condition)
            

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=3000)


# In[ ]:




