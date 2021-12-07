# import relevant modules
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,LeakyReLU
from keras.layers import Dropout
import numpy as np


#Spilit Data into Training and Test 
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

x_train=x_train/255
x_test=x_test/255

print(x_train)
print(y_train)

#For 3D and Colored Image
plt.imshow(x_train[0])
plt.show()

#Initialize ANN
classifier = Sequential()
classifier.add(Flatten(input_shape=[32, 32, 3]))

# adding Dropout layer
#classifier.add(Dropout(0.3))  # Ratio of drop - out between 0 to 1
# second Hidden Layer:
classifier.add(Dense(200 ,activation = "relu",))
 
# Adding second Drop - out layer :
classifier.add(Dropout(0.3))

# Output Layer : 
classifier.add(Dense(100,activation = 'softmax'))

classifier.compile(loss = "sparse_categorical_crossentropy",
                  optimizer= "adam",
                  metrics= ["accuracy"])
classifier.summary()

#Fitting the ANN to Training set
classifier.fit(x_train,y_train,epochs=10)

yp=classifier.predict(x_test)
print(yp)

print(classifier.evaluate(x_test,y_test))



