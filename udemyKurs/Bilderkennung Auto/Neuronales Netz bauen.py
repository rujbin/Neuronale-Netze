from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

print(y_train[4])
plt.imshow(X_train[4])
plt.show()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

y_train_car = y_train == 1
model.fit(X_train, y_train_car, batch_size=128, epochs=10, shuffle=True)

print(model.evaluate(X_train, y_train_car))

y_test_car = y_test == 1
print(model.evaluate(X_test, y_test_car))