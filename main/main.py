from shadow import create_dataset, three_D_model
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from learning_models import ThreeDCNN

shadow_iter = 3000; num_qubits = 4; num_simulations = 100

shadows, entropy = create_dataset(shadow_iter, num_qubits, num_simulations)

X_train, x_test, Y_train, y_test = train_test_split(shadows, entropy, test_size=0.33)

model = ThreeDCNN.model(shadows[0].shape[0], shadows[0].shape[1], shadows[0].shape[2])
training_set, validation_set = ThreeDCNN.preprocessing(X_train, x_test, Y_train, y_test)

# Compile model.
model.compile(loss="Poisson",optimizer=keras.optimizers.Adam())

epochs = 100

model.fit(training_set, validation_data=validation_set, epochs=epochs, shuffle=True)

