import tensorflow as tf
import keras
from keras import layers
import numpy as np
from sklearn.decomposition import KernelPCA

class CNNs(keras.layers.Layer):
  
  def threeDmodel(width, height, depth):
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=(width, height, 1), activation = 'relu')(inputs)
    x = layers.Conv3D(filters=64, kernel_size=(1, 1, depth), activation = 'relu')(inputs)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=(width, height, 1), activation = 'relu')(inputs)
    x = layers.Conv3D(filters=64, kernel_size=(1, 1, depth), activation = 'relu')(inputs)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=(width, height, 1), activation = 'relu')(inputs)
    x = layers.Conv3D(filters=64, kernel_size=(1, 1, depth), activation = 'relu')(inputs)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, output, name = "3dCNN")

    return model
  
  def twoDmodel(width, height):
    inputs = keras.Input((width, height, 1))

    x = layers.Conv2D(filters=16, kernel_size=(width, height), activation = 'relu', padding = "valid")(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(width, height), activation = 'relu', padding = "valid")(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=16, kernel_size=(width, height), activation = 'relu', padding = "valid")(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, output, name = "2dCNN")

    return model
  
  def kpca_fit(input_4D_array):

    kernel_pca = KernelPCA(n_components=1000)

    training_X = []

    for simulation in input_4D_array:
      simulation = np.transpose(simulation, (1, 0, 2))
      simulation_flattened = []
      for iteration in simulation: 
          iteration = np.array(iteration).flatten()
          simulation_flattened.append(iteration)
    
      kernel_pca.fit(np.array(simulation_flattened))
      training_X.append(kernel_pca.transform(np.array(simulation_flattened)))

    return training_X

  
  def preprocessing3D(X_train, x_test, Y_train, y_test):
    """
    Transfroms a sklearn-based train-test split to a 5D tensor required for fitting to 3d CNN models

    Args:
        - X_train, x_test: Data required to be learned and validated against
        - Y_train, y_test: Targets
        
    Returns:
        - 5D tensor of dimensions (None, input_dim, 1)
    """

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def dimensionality_change(volume, label):
        volume = tf.expand_dims(volume, axis=3)
        return volume, label

    train_dataset = (train_loader.map(dimensionality_change).batch(1))

    validation_dataset = (validation_loader.map(dimensionality_change).batch(1))

    return train_dataset, validation_dataset
  
  



  