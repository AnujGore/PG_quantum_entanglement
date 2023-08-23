from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from learning_models import *
import numpy as np
import matplotlib.pyplot as plt
import scipy


import tracemalloc
import time

start = time.time()
start_mem = tracemalloc.start()

shadow_iter = 320; num_qubits = 2; num_simulations = 1000

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

split = 0.45

activator_predef = lambda x: keras.backend.relu(x) + keras.backend.softplus(x)


X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=split)

X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)



# #For 3d CNN
# training_set_3d, validation_set_3d = CNNs.preprocessing3D(X_train, x_test, Y_train, y_test)
# model_3D = CNNs.threeDmodel(shadows[0].shape[0], shadows[0].shape[1], shadows[0].shape[2], activator_predef)




# # KPCA reduction
# training_X_K = CNNs.kpca_fit(shadows)

# X_train_K, x_test_K, Y_train_K, y_test_K = train_test_split(training_X_K, entropy, test_size=split)

# X_train_K = np.array(X_train_K)
# x_test_K = np.array(x_test_K)
# Y_train_K = np.array(Y_train_K)
# y_test_K = np.array(y_test_K)

# model_KPCA = CNNs.twoDmodel(np.shape(X_train_K)[1], np.shape(X_train_K)[2], activator_predef)




# #LDA reduction
# training_X_Diff = CNNs.diffusion_map(shadows)

# X_train_diff, x_test_diff, Y_train_diff, y_test_diff = train_test_split(training_X_Diff, entropy, test_size=split)

# X_train_diff = np.array(X_train_diff)
# x_test_diff = np.array(x_test_diff)
# Y_train_diff = np.array(Y_train_diff)
# y_test_diff = np.array(y_test_diff)



# model_LDA = CNNs.twoDmodel(np.shape(X_train_diff)[1], np.shape(X_train_diff)[2], activator_predef)

#Simple ANN


activation = "relu"
activation_2 = "softplus"

model = keras.Sequential([keras.layers.Flatten(), 
                          keras.layers.Dense(512, activation = activation), 
                          keras.layers.Dense(2, activation = activation_2, name = "secondLayer")
                        ])


# # #GRU + ANN

# shadows = np.transpose(shadows, (0, 2, 3, 1))

# def eigenvalues(array2D):
#     return scipy.linalg.svdvals(array2D)

# shadows_eigen = [[eigenvalues(array) for array in shadow_iter] for shadow_iter in shadows]

# X_train, x_test, Y_train, y_test = train_test_split(shadows_eigen, entropy, test_size=split)

# X_train = np.array(X_train)
# x_test = np.array(x_test)
# Y_train = np.array(Y_train)
# y_test = np.array(y_test)

# shadow_iter_shape = np.shape(shadows_eigen[0])
# shadow_full_shape = np.shape(shadows_eigen)

# inputs = keras.Input(shape = shadow_iter_shape)
# x = keras.layers.GRU(shadow_iter)(inputs)
# outputs = keras.layers.Dense(1, activation = activator_predef)(x)

# model = keras.Model(inputs = inputs, outputs = outputs)

# model.summary()

# Compile model.

epochs = 40

model.compile(loss="MeanSquaredLogarithmicError",optimizer=keras.optimizers.Adam())
history = model.fit(X_train, Y_train, epochs = epochs, validation_data= (x_test, y_test))

# # model_3D.compile(loss="MeanSquaredLogarithmicError",optimizer=keras.optimizers.Adam())
# # history_3D = model_3D.fit(training_set_3d, validation_data=validation_set_3d, epochs=epochs, shuffle=True)

# # model_KPCA.compile(loss="MeanSquaredError",optimizer=keras.optimizers.Adam())
# # history_KPCA = model_KPCA.fit(X_train_K, Y_train_K, epochs = epochs, validation_data= (x_test_K, y_test_K))

# # model_LDA.compile(loss="MeanSquaredError",optimizer=keras.optimizers.Adam())
# # history_LDA = model_LDA.fit(X_train_diff, Y_train_diff, epochs = epochs, validation_data= (x_test_diff, y_test_diff))

training_loss = history.history['loss']
test_loss = history.history['val_loss']

# # training_loss_3D = history_3D.history['loss']
# # test_loss_3D = history_3D.history['val_loss']

# # training_loss_KPCA = history_KPCA.history['loss']
# # test_loss_KPCA = history_KPCA.history['val_loss']

# # training_loss_LDA = history_LDA.history['loss']
# # test_loss_LDA = history_LDA.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'r--', label = "ANN (Training)")
plt.plot(epoch_count, test_loss, 'r-', label = "ANN (Testing)")
# plt.plot(epoch_count, training_loss_3D, 'b--', label = "3dConv (training)")
# plt.plot(epoch_count, test_loss_3D, 'b-', label = "3dConv (testing)")
# plt.plot(epoch_count, training_loss_KPCA, 'g--', label = "KPCA DimRed (training)")
# plt.plot(epoch_count, test_loss_KPCA, 'g-', label = "KPCA DimRed (testing)")
# plt.plot(epoch_count, training_loss_LDA, 'm--', label = "Diff DimRed (training)")
# plt.plot(epoch_count, test_loss_LDA, 'm-', label = "Diff DimRed (testing)")
plt.legend(loc = "best")
plt.xlabel('Epoch')
plt.ylabel('Loss (Logarithmic Loss)')
plt.title("Logarithmic Loss vs Epochs; Split = %.2f" %(split))

plt.show()

end = time.time()

# print("Training Loss: ", np.mean(training_loss)*100)
# print("Test Loss: ", np.mean(test_loss)*100)
# print("Time taken: ", end-start)
# print("Parameters used: ", model.count_params())
# print("Memory usage: ", tracemalloc.get_tracemalloc_memory())

tracemalloc.stop()