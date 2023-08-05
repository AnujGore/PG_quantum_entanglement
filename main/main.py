from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from learning_models import *
import numpy as np

shadow_iter = 10; num_qubits = 2; num_simulations = 3

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=0.33)

X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)

# # For 3d CNN
# training_set_3d, validation_set_3d = CNNs.preprocessing3D(X_train, x_test, Y_train, y_test)
# model = CNNs.threeDmodel(shadows[0].shape[0], shadows[0].shape[1], shadows[0].shape[2])

# #KPCA reduction
# training_X = CNNs.kpca_fit(shadows)
# model = CNNs.twoDmodel(np.shape(X_train)[1], np.shape(X_train)[2])


# #LDA reduction

# training_X = CNNs.diffusion_map(shadows)
# model = CNNs.twoDmodel(np.shape(X_train)[1], np.shape(X_train)[2])


# # For axial convolutions

# X_train_1 = X_train
# X_train = map(CNNs.axial_preprocessing, X_train)
# X_train = list(X_train)

# x_test = map(CNNs.axial_preprocessing, x_test)
# x_test = list(x_test)

# x_aligned_train = [X_train[i][0] for i in range(len(X_train))]
# y_aligned_train = [X_train[i][1] for i in range(len(X_train))]
# z_aligned_train = [X_train[i][2] for i in range(len(X_train))]

# x_aligned_test = [x_test[i][0] for i in range(len(x_test))]
# y_aligned_test = [x_test[i][1] for i in range(len(x_test))]
# z_aligned_test = [x_test[i][2] for i in range(len(x_test))]

# x_model = CNNs.twoDmodel_strided((num_qubits*4), shadow_iter, (num_qubits, shadow_iter), (num_qubits, 1))
# y_model = CNNs.twoDmodel_strided((num_qubits), (shadow_iter*4), (num_qubits, shadow_iter), (1, shadow_iter))
# z_model = CNNs.twoDmodel_strided(4, (shadow_iter*num_qubits), (num_qubits, 4), (1, 4))

# x_model.compile(loss="BinaryCrossentropy",optimizer=keras.optimizers.Adam())
# y_model.compile(loss="BinaryCrossentropy",optimizer=keras.optimizers.Adam())
# z_model.compile(loss="BinaryCrossentropy",optimizer=keras.optimizers.Adam())
# x_model.fit(x_aligned_train)
# y_model.fit(y_aligned_train)
# z_model.fit(z_aligned_train)

# model = keras.Sequential()
# model.add(x_model)
# model.add(y_model)
# model.add(z_model)
# model.add(keras.layers.Dense(256, activation = 'sigmoid'))
# model.add(keras.layers.Dense(128, activation = 'sigmoid'))
# model.add(keras.layers.Dense(10, activation = 'sigmoid'))


# Compile model.

epochs = 10

model.compile(loss="BinaryCrossentropy",optimizer=keras.optimizers.Adam())
model.fit(X_train, Y_train, epochs = epochs, validation_data= (x_test, y_test))
loss = model.evaluate(x_test, y_test)



print("Loss", loss)







