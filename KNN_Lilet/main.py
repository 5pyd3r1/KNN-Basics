import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

path = "../Dataset/iris.csv"
data = pd.read_csv(path, delimiter=',')

# Was vorhergesagt wird
col_name = 'species'

# Spalte species in int Form umwandeln
data[col_name] = data[col_name].astype('category')
data[col_name] = data[col_name].cat.codes

# Aufteilung der Tabellen (Input = data, Output = col)
col = data[col_name]
data = data.drop([col_name], axis=1)

# Aus den zwei Tabellen 4 Tabellen erzeugen für Train und Test Phase
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2)

# Modell des KNN (1. Hidden layer 32 Knoten 2. Hidden layer 64 Knoten, 3. Output layer mit 3 Knoten)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.sigmoid, input_dim=4),
    tf.keras.layers.Dense(64, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# Konfiguration des Lernprozesses
model.compile(
    optimizer='adam',   # Optimierungsfunktion adam
    loss='sparse_categorical_crossentropy',     # Verlustfunktion
    metrics=['accuracy']    # Anzahl richtiger Vorhersagen / Gesamtanzahl Vorhersagen
)

# 30 Durchlaueufe
model.fit(train_data, train_col, epochs=60)

# Test Phase
test_loss, test_acc = model.evaluate(test_data, test_col)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)