import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

path = "../Dataset/adult-2.csv"
data = pd.read_csv(path, delimiter=';')

# Correlation detection
'''
correlation = data[data.columns].corr(numeric_only=True)
print(correlation)
print("All Correlations")
print("-" * 30)
correlation_abs_sum = correlation[correlation.columns].abs().sum()
print(correlation_abs_sum)
print("Weakest Correlations")
print("-" * 30)
print(correlation_abs_sum.nsmallest(5))
'''

data.drop(['fnlwgt', 'capital.loss', 'capital.gain'], axis=1, inplace=True)

# Income prediction
col = data['income']
col = pd.get_dummies(col, dtype=float) # 01 10 -> Encoding for below or above 50k

data = data.drop(['income'], axis=1)

# Converting String columns to int
conv_num = [
    'workclass', 'education', 'marital.status', 'occuaption', 'relationship',
    'hours.per.week', 'native.country'
]
data[conv_num] = data[conv_num].astype('category')
data[conv_num] = data[conv_num].apply(lambda x: x.cat.codes)

# OHE for the rest of the columns
conv_ohe = ['race', 'sex']
data = pd.get_dummies(data, columns=conv_ohe, dtype=float)

# Splitting the data
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=data.shape[1]), # data.shape[1] = 13 -> 13 Columns
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Configuring the learning process
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
model.fit(train_data, train_col, epochs=18)

# Testing
test_loss, test_acc = model.evaluate(test_data, test_col)
print("Test accuracy:", test_acc)