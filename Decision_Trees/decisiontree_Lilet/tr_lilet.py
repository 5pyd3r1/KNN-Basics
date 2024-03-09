import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

path = "../../Dataset/iris.csv"
data = pd.read_csv(path, delimiter=",")

# What should be predicted
col_name = "species"

col = data[col_name]
data = data.drop([col_name], axis=1)

train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2)

# Tree creation
tr = tree.DecisionTreeClassifier()

# train
tr.fit(train_data, train_col)

# prediction
predicted_col = tr.predict(test_data)

# Rating
score = metrics.accuracy_score(test_col, predicted_col)
print(score)

# Plotting the tree
tree.plot_tree(tr)
plt.show()
