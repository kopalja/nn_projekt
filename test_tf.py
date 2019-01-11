import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("data/dogs_and_snakes.pkl", 'rb') as f:
    training_data, validation_data = pickle.load(f, encoding='latin1')

height, width = 150, 200
train_size = 5000
valid_size = 500

x_train, y_train = training_data
x_valid, y_valid = validation_data
x_train = np.asarray([np.reshape(im, (height, width, 1)) for im in x_train[:train_size]])
x_valid = np.asarray([np.reshape(im, (height, width, 1)) for im in x_valid[:valid_size]])
y_train = np.asarray(y_train[:train_size])
y_valid = np.asarray(y_valid[:valid_size])
print(x_train.shape)
print(y_train.shape)
#plt.imshow(x_train[10], cmap='gray')

print(f'Length of training data: {len(x_train)}')

model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape = [height, width, 1]))

model.add(keras.layers.Conv2D(filters = 20, kernel_size = 5, strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 40, kernel_size = 5, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 70, kernel_size = 5, padding='same', strides= 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation = 'sigmoid'))
model.add(keras.layers.Dense(2, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(100):
    print(f"Starting epoch {i}")
    history = model.fit(x = x_train, y = y_train, epochs=1, batch_size=32) 

    #loss and accuracy for testing data
    res = model.evaluate(x_valid, y_valid)
    print(f"Validation loss = {res[0]}")
    print(f"Validation accuracy = {res[1]}")
