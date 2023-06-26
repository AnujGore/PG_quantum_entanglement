from shadow import create_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

dataset = create_dataset(50, 2, 1)

dataset_X = []
dataset_Y = []

for data in dataset:
    dataset_X.append(data[0])
    dataset_Y.append(data[1])

data_see = sum(dataset_X[0])
data_see = (data_see + 1)/2

print(dataset[-1][-1])

plt.imshow(data_see, cmap='Greys')
plt.show()


# X_train, X_test, Y_train, Y_test = train_test_split(dataset_X, dataset_Y, random_state=1)

# regr = MLPRegressor(random_state=1).fit(X_train, Y_train)
# score = regr.score(X_test, Y_test)
# print(score)


