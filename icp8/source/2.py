import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd

# load dataset
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("Breas Cancer.csv")
#dataset=dataset.drop('fractal_dimension_worst',axis=1)
x1=dataset.drop(['diagnosis','fractal_dimension_worst'],axis=1)
y1=dataset['diagnosis']
y1=y1.replace(['M','B'],[1,0])
#print(y1)

import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(x1, y1,
                                                    test_size=0.3, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(100, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(9,  activation='sigmoid')) # hidden layer

my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,verbose=0,initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test,verbose=0))