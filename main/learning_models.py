import tensorflow as tf
import keras
from keras import layers
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

class CNNs(keras.layers.Layer):
  
  def threeDmodel(width, height, depth):
    """
    Three dimensional CNN for feature extraction. 

    Model is built sequentially with Conv2D (x, y, 1) strides + Conv1D (1, 1, z) strides + AveragePooling2D + Normalization repeated 3 times.

    Inputs:
       - width: Width of filter
       - height: Height of filter
       - depth: Depth of the filter

    Returns:
       - Model (keras.Model)
       
    """
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
    """
    Two dimensional CNN for feature extraction. 

    Model is built sequentially with Conv2D + AveragePooling2D + Normalization repeated 3 times.

    Inputs:
       - width: Width of filter
       - height: Height of filter

    Returns:
       - Model (keras.Model)

    """
   #  inputs = tf.convert_to_tensor((width, height, 1), dtype = tf.float32)
    inputs = tf.expand_dims((width, height, 1), axis = 0)
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
    """
    Function which returns the diffusion map of a 4D array as a 3D array (dimensionality reduction) via Kernel PCA.

    Args:
       - input_4D_array (np.array 4 dimensions): 4 dimensional array of choice. For this program, the dimensions are simulation x pauli basis x shadow iterations x number of qubits.

    Returns:
       - np.array 3 dimensions: For this program it would be simulation x shadow iterations x (pauli basis and number of qubits).
    
    """
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
  
  def diffusion_map(input_4D_array):
    """
    Function which returns the diffusion map of a 4D array as a 3D array (dimensionality reduction) via IsoMapping

    Args:
       - input_4D_array (np.array 4 dimensions): 4 dimensional array of choice. For this program, the dimensions are simulation x pauli basis x shadow iterations x number of qubits.

    Returns:
       - np.array 3 dimensions: For this program it would be simulation x shadow iterations x (pauli basis and number of qubits).
    
    """

    diff_maps = Isomap(n_components=4)

    training_X = []

    for simulation in input_4D_array:
      simulation = np.transpose(simulation, (1, 0, 2))
      simulation_flattened = []
      for iteration in simulation: 
          iteration = np.array(iteration).flatten()
          simulation_flattened.append(iteration)
    
      diff_maps.fit(np.array(simulation_flattened))
      training_X.append(diff_maps.transform(np.array(simulation_flattened)))

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
  
  def axial_preprocessing(input_3d_array):
     
     input_data = [data for data in input_3d_array]
     
     x_aligned = np.concatenate(input_data, axis = 1)
     y_aligned = np.concatenate(input_data, axis = 0)

     input_data = [data for data in np.transpose(input_3d_array, (1, 0, 2))]
     z_aligned = np.concatenate(input_data, axis = 0)

     return (x_aligned, y_aligned, z_aligned)
  
  def twoDmodel_strided(width, height, kernel_size, strides):
    """
    Two dimensional CNN for feature extraction. 

    Model is built sequentially with Conv2D + AveragePooling2D + Normalization repeated 3 times.

    Inputs:
       - width: Width of filter
       - height: Height of filter

    Returns:
       - Model (keras.Model)

    """
    inputs = keras.Input((width, height, 1))

    x = layers.Conv2D(filters=16, kernel_size=kernel_size, strides = strides, activation = 'relu', padding = "valid")(inputs)
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
  

     





  



  