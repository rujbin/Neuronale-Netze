import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# 1. Datensatz laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Daten normalisieren
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# 3. Modell kompilieren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Modell trainieren
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# 5. Modell evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# 6. Vorhersagen treffen
predictions = model.predict(x_test)

# Beispielvorhersage für das erste Bild im Testdatensatz
print(f'Vorhergesagte Ziffer: {np.argmax(predictions[0])}')

# 7. Visualisierung der Ergebnisse (optional)
plt.imshow(x_test[0], cmap='gray')
plt.title(f'Vorhergesagte Ziffer: {np.argmax(predictions[0])}')
plt.show()
