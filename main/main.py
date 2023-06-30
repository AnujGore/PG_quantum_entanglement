from shadow import create_dataset
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt

dataset = create_dataset(10, 2, 1)

dataset_X = []
dataset_Y = []

for data in dataset:
    dataset_X.append(data[0])
    dataset_Y.append(data[1])

print(dataset_X)
print(dataset_Y)




