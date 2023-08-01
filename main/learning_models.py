import tensorflow as tf
import keras
from keras import layers

class ThreeDCNN(keras.layers.Layer):
  
  def model(width, height, depth):
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
  
  def preprocessing(X_train, x_test, Y_train, y_test):
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
  
  



  