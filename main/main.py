from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from learning_models import *
import numpy as np
import matplotlib.pyplot as plt

import tracemalloc
import time

start = time.time()
start_mem = tracemalloc.start()

# def ln_activation(x):
#      return np.sinh(x-1) - np.cosh(x-1) + 1

# np_ln_activation = np.vectorize(ln_activation)

# np_ln_activation_f = lambda x: np_ln_activation(x).astype(np.float32)

# def tf_ln_activation(x, name = None):
#     with tf.name_scope("naturalLog") as name:
#         y = tf.py_function(np_ln_activation_f,
#                            [x],
#                            [tf.float32],
#                            name = name)
#         return tf.convert_to_tensor(y[0])


shadow_iter = 320; num_qubits = 2; num_simulations = 1000

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

split = 0.3

activator_predef = lambda x: keras.backend.relu(x) + keras.backend.softplus(x)


# X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=split)

# X_train = np.array(X_train)
# x_test = np.array(x_test)
# Y_train = np.array(Y_train)
# y_test = np.array(y_test)

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


#LDA reduction
training_X_Diff = CNNs.diffusion_map(shadows)

X_train_diff, x_test_diff, Y_train_diff, y_test_diff = train_test_split(training_X_Diff, entropy, test_size=split)

X_train_diff = np.array(X_train_diff)
x_test_diff = np.array(x_test_diff)
Y_train_diff = np.array(Y_train_diff)
y_test_diff = np.array(y_test_diff)

model_LDA = CNNs.twoDmodel(np.shape(X_train_diff)[1], np.shape(X_train_diff)[2], activator_predef)

# #Simple ANN

# def gaussian(x):
#     return keras.backend.exp(-keras.backend.pow(x,2))

# def natural_log(x):
#     return gaussian(x) + keras.backend.tanh(x-1) - keras.backend.sigmoid(x)


activation = "relu"
activation_2 = "softplus"
# activation_3 = gaussian
# activation_4 = tf_ln_activation

# model = keras.Sequential([keras.layers.Flatten(), 
#                           # keras.layers.Dense(1024, activation = activation), 
#                           keras.layers.Dense(512, activation = activation), 
#                           # keras.layers.Dense(512, name = "firstLayer"),
#                           # keras.layers.Activation(tf_ln_activation, name = "activationLayer"),
#                           # keras.layers.Dense(256, activation = activation), 
#                           # keras.layers.Dense(128, activation = activation_4),
#                           # keras.layers.Dense(64, activation = activation),
#                           keras.layers.Dense(2, activation = activation_2, name = "secondLayer")
#                         ])



# Compile model.

epochs = 100
activator_s = "Self"
activator_pre = "Predefined"

# model.compile(loss="MeanSquaredLogarithmicError",optimizer=keras.optimizers.Adam())
# history = model.fit(X_train, Y_train, epochs = epochs, validation_data= (x_test, y_test))

# model_3D.compile(loss="MeanSquaredLogarithmicError",optimizer=keras.optimizers.Adam())
# history_3D = model_3D.fit(training_set_3d, validation_data=validation_set_3d, epochs=epochs, shuffle=True)

# model_KPCA.compile(loss="MeanSquaredLogarithmicError",optimizer=keras.optimizers.Adam())
# history_KPCA = model_KPCA.fit(X_train_K, Y_train_K, epochs = epochs, validation_data= (x_test_K, y_test_K))

model_LDA.compile(loss="MeanSquaredError",optimizer=keras.optimizers.Adam())
history_LDA = model_LDA.fit(X_train_diff, Y_train_diff, epochs = epochs, validation_data= (x_test_diff, y_test_diff))

# training_loss = history.history['loss']
# test_loss = history.history['val_loss']

# training_loss_3D = history_3D.history['loss']
# test_loss_3D = history_3D.history['val_loss']

# training_loss_KPCA = history_KPCA.history['loss']
# test_loss_KPCA = history_KPCA.history['val_loss']

training_loss_LDA = history_LDA.history['loss']
test_loss_LDA = history_LDA.history['val_loss']

epoch_count = range(1, len(training_loss_LDA) + 1)

# plt.plot(epoch_count, training_loss, 'r--', label = "ANN (Training)")
# plt.plot(epoch_count, test_loss, 'b-', label = "ANN (Testing)")
# plt.plot(epoch_count, training_loss_3D, 'b--', label = "3dConv (training)")
# plt.plot(epoch_count, test_loss_3D, 'b-', label = "3dConv (testing)")
# plt.plot(epoch_count, training_loss_KPCA, 'g--', label = "KPCA DimRed (training)")
# plt.plot(epoch_count, test_loss_KPCA, 'g-', label = "KPCA DimRed (testing)")
plt.plot(epoch_count, training_loss_LDA, 'm--', label = "Diff DimRed (training)")
plt.plot(epoch_count, test_loss_LDA, 'm-', label = "Diff DimRed (testing)")
plt.legend(loc = "best")
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error Loss)')
plt.title("Mean Squared Error Loss vs Epochs; Split = %.2f; Activator = %s" %(split, activator_pre))

plt.show()

# end = time.time()

# print("Training Loss: ", np.mean(training_loss_LDA)*100)
# print("Test Loss: ", np.mean(test_loss_LDA)*100)
# print("Time taken: ", end-start)
# print("Memory usage: ", tracemalloc.get_tracemalloc_memory())
# print("Parameters used: ", model_LDA.count_params())
# tracemalloc.stop()

# # plt.text(30, max(training_loss), "Memory usage: ")


# # print("Size of model: ", model_LDA.count_params())