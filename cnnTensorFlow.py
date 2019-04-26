from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.load("data/x_train.npy")
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
y_train = np.load("data/y_train.npy")

X_test = np.load("data/x_test.npy")
X_test = (X_test - np.mean(X_test, axis=0))/np.std(X_test, axis=0)
y_test = np.load("data/y_test.npy")
#---------------------------------------------
X_train = X_train.reshape(50000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''
The code above simply loads the data and one-hot encode it. We load the MNIST
data from the files given to us.
'''
#---------------------------------------------

#create model
model = Sequential()
# allows us to create model step by step

#add model layers
'''
Kernel size is typically 3 or 5. Experimenting with a value of 4 didn't decrease the accuracy significantly,
i.e. only by like 1% or so.
'''
model.add(Conv2D(64, kernel_size=5, activation='tanh', input_shape=(28,28,1))) # Adds a convolution layer
'''
Each model.add(Conv2D...) essentially adds another layer to the neural network. Instead of using an affine forward
to do so, we depend on a convolution instead, taking a square cluster of kernel_size, and looping through the image.
At each iteration of the for loop, we take the convolution of the feature (the square cluster) and the part of the image
that we're on. Once we do this, we can get a 2D array using the answers from each convolution, based on where in the
image each patch is located.
'''

model.add(Conv2D(32, kernel_size=3, activation='tanh'))
model.add(Flatten()) # Turns 2d array into 1d array
model.add(Dense(10, activation='softmax')) # condenses layers down into final 10 classes


# compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''
Optimizer: 'Adam' an optimizer that adjusts the learning rate while training
loss: categorical_crossentropy most commonly used in classification
metrics: prints accuracy of CNN
'''
#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)
# Runs the model
