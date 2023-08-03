from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from learning_models import CNNs
import numpy as np

shadow_iter = 30; num_qubits = 2; num_simulations = 3

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

#For 3d CNN
# X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=0.33)

# model3D= CNNs.threeDmodel(shadows[0].shape[0], shadows[0].shape[1], shadows[0].shape[2])
# training_set_3d, validation_set_3d = CNNs.preprocessing(X_train, x_test, Y_train, y_test)


#KPCA reduction
training_X = CNNs.kpca_fit(shadows)

X_train, x_test, Y_train, y_test = train_test_split(training_X, entropy, test_size=0.33)

X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)


model2d = CNNs.twoDmodel(np.shape(X_train)[1], np.shape(X_train)[2])

# Compile model.

epochs = 10

model2d.compile(loss="Poisson",optimizer=keras.optimizers.Adam())
model2d.fit(X_train, Y_train, epochs = epochs, validation_data= (x_test, y_test))

