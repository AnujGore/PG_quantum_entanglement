from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from learning_models import *
import numpy as np
import matplotlib.pyplot as plt


shadow_iter = 50; num_qubits = 2; num_simulations = 1000

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=0.2)

X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)

# #For 3d CNN
# training_set_3d, validation_set_3d = CNNs.preprocessing3D(X_train, x_test, Y_train, y_test)
# model = CNNs.threeDmodel(shadows[0].shape[0], shadows[0].shape[1], shadows[0].shape[2])

# #KPCA reduction
# training_X = CNNs.kpca_fit(shadows)
# model = CNNs.twoDmodel(np.shape(X_train)[1], np.shape(X_train)[2])


# #LDA reduction

# training_X = CNNs.diffusion_map(shadows)
# model = CNNs.twoDmodel(np.shape(X_train)[1], np.shape(X_train)[2])

#Simple ANN

def gaussian(x):
    return keras.backend.exp(-keras.backend.pow(x,2))

def natural_log(x):
    return gaussian(x) + keras.backend.tanh(x-1) - keras.backend.sigmoid(x)


activation = "relu"
activation_2 = "softplus"
activation_3 = gaussian
activation_4 = natural_log

model = keras.Sequential([keras.layers.Flatten(), 
                        #   keras.layers.Dense(2048, activation = activation), 
                        #   keras.layers.Dense(1024, activation = activation), 
                          keras.layers.Dense(256, activation = activation), 
                          keras.layers.Dense(64, activation = activation),
                          keras.layers.Dense(2, activation = activation_2)])


# Compile model.

epochs = 100

model.compile(loss="MeanSquaredError",optimizer=keras.optimizers.Adam())
history = model.fit(X_train, Y_train, epochs = epochs, validation_data= (x_test, y_test))
loss = model.evaluate(x_test, y_test)

training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Mean Squared Error Loss vs Epochs")
plt.show()