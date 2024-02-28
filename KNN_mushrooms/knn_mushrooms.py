import tensorflow as tf
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

path = "../Dataset/mushrooms.csv"
data = pd.read_csv(path, delimiter=',')
# print(data.head())

# Numbers of empty cells
# print(data.columns[data.isnull().any()])

# Predicted value
col_name = 'class'
col = pd.get_dummies(data[col_name], dtype=float)  # 01 10 -> Encoding for poisonous or edible
data = data.drop([col_name], axis=1)

le = LabelEncoder()
data = data.apply(le.fit_transform)

# Correlation detection
correlation = data[data.columns].corr(numeric_only=True)
print(correlation)
print("All Correlations")
print("-" * 30)
correlation_abs_sum = correlation[correlation.columns].abs().sum()
print(correlation_abs_sum)
print("Weakest Correlations")
print("-" * 30)
print(correlation_abs_sum.nsmallest(5))

# Drop the columns with the weakest correlation
data.drop(['veil-type', 'cap-shape', 'cap-surface', 'habitat', 'veil-color'], axis=1, inplace=True)


# Scaler
s_scaler = StandardScaler()
data = s_scaler.fit_transform(data)

# Splitting the data
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=data.shape[1]),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
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
model.fit(train_data, train_col, epochs=20)

# Test Phase
test_loss, test_acc = model.evaluate(test_data, test_col)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# Save the model, scale and label encoder
# model.save('mushrooms_model.h5')
# joblib.dump(s_scaler, 'mushrooms_scaler.pkl')
# joblib.dump(le, 'mushrooms_le.pkl')

# Load the model, scale and label encoder
# model = tf.keras.models.load_model('mushrooms_model.h5')
# s_scaler = joblib.load('mushrooms_scaler.pkl')
# le = joblib.load('mushrooms_le.pkl')

