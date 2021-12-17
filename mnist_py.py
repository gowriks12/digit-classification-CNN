import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from keras import backend
import matplotlib.pyplot as plt

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train)

# no hidden layers
model1 = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10)
])

model1.compile(optimizer='SGD',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])
backend.set_value(model1.optimizer.learning_rate, 0.25)

model1_fit = model1.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model1.summary()

# One hidden layer of 7 neurons
model2 = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(7, activation='sigmoid'),
    Dense(10)
])

model2.compile(optimizer='SGD',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])
backend.set_value(model2.optimizer.learning_rate, 0.25)

model2_fit = model2.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model2.summary()

# One hidden layer with 49 neurons
model3 = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(49, activation='sigmoid'),
    Dense(10)
])

model3.compile(optimizer='SGD',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])
backend.set_value(model3.optimizer.learning_rate, 0.25)

model3_fit = model3.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model3.summary()

val1_loss   = model1_fit.history['val_loss']
val2_loss   = model2_fit.history['val_loss']
val3_loss   = model3_fit.history['val_loss']

xc         = range(10)

plt.figure()
plt.plot(xc, val1_loss,'red')
plt.plot(xc, val2_loss,'blue')
plt.plot(xc, val3_loss,'green')
plt.show()