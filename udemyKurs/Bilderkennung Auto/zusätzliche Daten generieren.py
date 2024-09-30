from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
import numpy as np

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

print(y_train[4])
plt.imshow(X_train[4])
plt.show()

gen = ImageDataGenerator(width_shift_range=3, height_shift_range=3, zoom_range=0.1, horizontal_flip=True)
for batch in gen.flow(X_train, y_train, shuffle=False):
    first_image = batch[0][0]
    plt.imshow(first_image)
    plt.show()
    break

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32,32,3), activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))
model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

y_train_car = y_train == 1
model.fit(X_train, y_train_car, batch_size=128, epochs=10, shuffle=True)

print(model.evaluate(X_train, y_train_car))

y_test_car = y_test == 1
print(model.evaluate(X_test, y_test_car))