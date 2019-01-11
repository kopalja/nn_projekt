import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("data/dogs_and_snakes.pkl", 'rb') as f:
    training_data, validation_data = pickle.load(f, encoding='latin1')

height, width = 150, 200
train_size = 5000
valid_size = 1000

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
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Conv2D(filters = 40, kernel_size = 5, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Conv2D(filters = 70, kernel_size = 5, padding='same', strides= 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation = 'sigmoid', activity_regularizer = keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(2, activation = 'softmax'))

model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

for i in range(100):
    print(f"Starting epoch {i + 1}")
    perf_train = model.fit(x = x_train, y = y_train, epochs=1, batch_size=32) 
    train_loss.append(perf_train.history['loss'])
    train_acc.append(perf_train.history['acc'])

    #loss and accuracy for testing data
    perf_valid = model.evaluate(x_valid, y_valid)
    valid_loss.append(perf_valid[0])
    valid_acc.append(perf_valid[1])
    print(f"Validation loss = {perf_valid[0] :1.4}")
    print(f"Validation accuracy = {perf_valid[1] :1.4}")


plt.plot(train_loss)
plt.plot(train_acc)
plt.show()

plt.plot(valid_loss)
plt.plot(valid_acc)
plt.show()
