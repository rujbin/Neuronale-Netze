# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/
# FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

import gzip
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)
    
X_train = open_images("../data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("../data/fashion/train-labels-idx1-ubyte.gz")

X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_labels("../data/fashion/t10k-labels-idx1-ubyte.gz")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modell 1
model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train.reshape(60000, 28, 28, 1), y_train, epochs=10, batch_size=1000)

# Modell 2, Transfer der Gewichte aus Modell 1
model2 = Sequential()
model2.add(Conv2D(10,
                  kernel_size=(3, 3),
                  activation="sigmoid",
                  input_shape=(28,28,1)))
model2.layers[0].set_weights(model.layers[0].get_weights())

result = model2.predict(X_test[0].reshape(1, 28, 28, 1))

# Anzeige des Ergebnisses
plt.imshow(result[0][:, :, 6])
plt.show()

# Modell 3
model3 = Sequential()
model3.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
model3.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid"))
model3.add(Flatten())
model3.add(Dense(10, activation="softmax"))

model3.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model3.fit(X_train.reshape(60000, 28, 28, 1), y_train, epochs=1, batch_size=1000)
