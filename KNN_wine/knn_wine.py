import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

path = "..\Dataset\winequality-red.csv"
data = pd.read_csv(path, delimiter=';')

# What is predicted
col_name = 'quality'

# Convert the column 'quality' to int
data[col_name] = data[col_name].astype('category')
data[col_name] = data[col_name].cat.codes

# Splitting the tables (Input = data, Output = col)
col = data[col_name]
data = data.drop([col_name], axis=1)

# Splitting the data into 4 tables for Train and Test Phase
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2)

# Model of the KNN (1. Hidden layer 32 nodes 2. Hidden layer 120 nodes, 3. Hidden layer 120 nodes, 4. OL with 11 nodes)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.sigmoid, input_dim=11),
    tf.keras.layers.Dense(120, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(120, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(11, activation=tf.nn.softmax)
])

# Configuring the learning process
model.compile(
    optimizer='adam',   # Optimization function adam
    loss='sparse_categorical_crossentropy',     # Loss function
    metrics=['accuracy']    # Number of correct predictions / Total number of predictions
)

# 70 runs
model.fit(train_data, train_col, epochs=70)

# Test Phase
test_loss, test_acc = model.evaluate(test_data, test_col)
print("Test accuracy:", test_acc)
