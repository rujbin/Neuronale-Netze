Natürlich! Hier ist eine README-Dokumentation für das CNN-Beispiel:

---

# CNN Beispiel für MNIST Klassifizierung

Dieses Projekt enthält ein einfaches Convolutional Neural Network (CNN), das mit TensorFlow und Keras erstellt wurde, um Bilder von handgeschriebenen Ziffern aus dem MNIST-Datensatz zu klassifizieren. 

## Installation

Stelle sicher, dass du die benötigten Pakete installiert hast. Du kannst dies tun, indem du die folgenden Befehle ausführst:

```bash
pip install tensorflow matplotlib
```
## Erklärungen

### 1. Daten vorbereiten

- **`datasets.mnist.load_data()`**: Lädt den MNIST-Datensatz, der handgeschriebene Ziffern enthält.
- **`reshape((60000, 28, 28, 1))`**: Ändert die Form der Trainingsbilder zu `(60000, 28, 28, 1)`, was 60.000 Bilder mit einer Größe von 28x28 Pixeln und einem Farbkanal (Graustufen) darstellt.
- **`astype('float32') / 255`**: Normalisiert die Pixelwerte, indem sie auf den Bereich [0, 1] skaliert werden.

### 2. Modell erstellen

- **`layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))`**: Der erste Convolutional Layer mit 32 Filtern der Größe 3x3, der ReLU-Aktivierungsfunktion und einer Eingabegröße von 28x28x1.
- **`layers.MaxPooling2D((2, 2))`**: Der Max-Pooling Layer reduziert die räumliche Dimension der Bilddarstellung, indem er das Maximum in jedem 2x2 Bereich auswählt.
- **`layers.Flatten()`**: Wandelt die 2D-Daten in einen 1D-Vektor um, der an die Dense Layer weitergegeben wird.
- **`layers.Dense(64, activation='relu')`**: Vollständig verbundene Schicht mit 64 Einheiten und ReLU-Aktivierung.
- **`layers.Dense(10)`**: Die Ausgabeschicht mit 10 Einheiten, die die 10 Ziffernklassen repräsentiert. Die Aktivierungsfunktion wird nicht angegeben, da `SparseCategoricalCrossentropy` verwendet wird.

### 3. Modell kompilieren

- **`optimizer='adam'`**: Der Adam-Optimizer wird verwendet, um die Gewichte des Modells zu aktualisieren.
- **`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`**: Die Verlustfunktion zur Berechnung der Differenz zwischen den vorhergesagten und tatsächlichen Klassen. `from_logits=True` bedeutet, dass die Ausgaben des Modells noch nicht durch eine Softmax-Funktion normalisiert wurden.

### 4. Modell trainieren

- **`model.fit()`**: Trainiert das Modell auf den Trainingsdaten und validiert es auf den Testdaten für 5 Epochen.

### 5. Modell bewerten und plotten

- **`model.evaluate()`**: Bewertet das Modell auf den Testdaten und gibt die Testgenauigkeit aus.
- **`plt.plot()`**: Plottet den Verlauf der Trainingsgenauigkeit und Validierungsgenauigkeit über die Epochen.

## Ergebnis

Nach dem Training des Modells wird die Testgenauigkeit auf der Konsole ausgegeben und der Trainingsverlauf wird als Diagramm angezeigt.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe [LICENSE](LICENSE) für Details.

---

Fühl dich frei, die README-Datei nach Bedarf anzupassen!
