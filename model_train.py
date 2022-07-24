import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# function for training the models
def train(model, epochs, batch_size, X_train, y_train, X_validation, y_validation):

    # training of the model
    history = model.fit(X_train, # training data
                        y_train, # labels
                        epochs=epochs, # epochs of training
                        batch_size=batch_size, # training batch size 
                        validation_data=(X_validation, y_validation)) # validation data and labels
    return history 


# function to create the RNN-LSTM model with the Tensorflow and Keras libraries.
def rnn_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.01):


    # Definition of a Keras Sequential Model as a basis 
    model = tf.keras.models.Sequential()

    # First RNN-LSTM layer in combination with dropout and batchnormalization as regularization techniques, all output sequneces are outputed
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, activation='tanh')) 
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.BatchNormalization())
    
    # Second RNN-LSTM layer in combination with dropout and batchnormalization as regularization techniques, all output sequneces are outputed
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, activation='tanh')) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
         
    # Third RNN-LSTM layer in combination with dropout and batchnormalization as regularization techniques, only last output sequence is outputed    
    model.add(tf.keras.layers.LSTM(128, activation='tanh')) 
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.BatchNormalization())

    # Dense Layer or Fully Connected Layer in combination with the dropout and batchnormalization regularization technique
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.BatchNormalization())
    
    # softmax output layer to output class probabilities of the 10 classes
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    optimiser = tf.optimizers.Adam(learning_rate=0.001) # Determination of the optimizer and learning rate for the training process

    # compile model
    model.compile(optimizer=optimiser,          # Optimizer
                  loss=loss,                    # Loss or error function
                  metrics=["accuracy"])         # Metric to be monitored during training, here accuracy

    # Output of the summary of the model  
    model.summary()

    return model


# Funktion zur Erstellung des CNN-Modells mit den Bibliotheken Tensorflow und Keras
def cnn_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.01):
    
    # Definition of a Keras Sequential Model as a basis 
    model = tf.keras.models.Sequential()

    # First Convolutional Layer in combination with Maxpooling Operation 
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))  # Angabe der Eingabegröße des ersten Layers notwendig
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same')) 

    # Second Convolutional Layer in combination with Maxpooling Operation  
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

    # Third Convolutional Layer in combination with Maxpooling Operation 
    model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

    # so called "flatten" i.e. flatten the output because Dense layer expects a one dimensional array
    model.add(tf.keras.layers.Flatten())
    
    # First Dense Layer or Fully Connected Layer in combination with the dropout regularization technique.
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    tf.keras.layers.Dropout(0.2) 
    
    # Second Dense Layer or Fully Connected Layer in combination with the dropout regularization technique.
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.2) 
    
    # softmax output layer to output class probabilities of the 10 classes
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate) # Determination of the optimizer and learning rate for the training process

    # Kompielierung des models
    model.compile(optimizer=optimiser,          # Optimizer
                  loss=loss,                    # Loss or error function
                  metrics=["accuracy"])         # Metric to be monitored during training, here accuracy

    # Output of the summary of the model   
    model.summary()

    return model


def plot_history(history):

    fig, axs = plt.subplots(2)
    
    # Display of the accuracy curve over the training process
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    
    # Display of the loss curve over the training process
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")
    
    fig.tight_layout()
    plt.show()
    
    
    