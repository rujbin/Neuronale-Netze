import gzip
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Funktionen zum Laden der Daten
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

# Laden der Fashion-MNIST-Daten
X_train = open_images("../data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("../data/fashion/train-labels-idx1-ubyte.gz")
X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_labels("../data/fashion/t10k-labels-idx1-ubyte.gz")

# One-Hot-Encoding der Labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN-Modell 1 (Max-Pooling, ReLU)
model1 = Sequential()
model1.add(Conv2D(10, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(10, activation="softmax"))

model1.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Modell 1 trainieren
model1.fit(
    X_train.reshape(60000, 28, 28, 1),
    y_train,
    epochs=10,
    batch_size=1000
)

# Modell 1 evaluieren
model1.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)

# CNN-Modell 2 (Max-Pooling, Sigmoid)
model2 = Sequential()
model2.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(10, activation="softmax"))

model2.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Modell 2 trainieren
model2.fit(
    X_train.reshape(60000, 28, 28, 1),
    y_train,
    epochs=10,
    batch_size=1000
)

# Modell 2 evaluieren
model2.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
