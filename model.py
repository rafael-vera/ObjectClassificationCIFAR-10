import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
import numpy as np

from file_paths import *

class DeepLearningModel():
    """
    Attributes
    ----------
    OPTIMIZERS: dict
        A dictionary of possible optmization algorithms to be use
    BATCH_SIZE: int
        The number of examples to be forwarded across the network
    NUM_CLASSES: int
        The number of total classes the data has
    EPOCHS: int
        The number of times the model will be trained
    
    Methods
    -------
    build_shallow_model(optimizer: str)
        Sets a shallow model architecture with a given optimizer
    build_deep_model(optimizer: str)
        Sets a deep model architecture with a given optimizer
    train_model()
        The current model is trained and evaluated showing the training progress
    save_model(path: str)
        The current model is saved in the given path
    """
    
    OPTIMIZERS = {
        'Adadelta': tf.keras.optimizers.Adadelta(),
        'SGD': tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    }
    
    BATCH_SIZE = 64
    NUM_CLASSES = 10
    EPOCHS = 60
    
    def __init__(self) -> None:
        """The train and test data are loaded
        """
        
        img_rows, img_cols = 32, 32
        
        self.x_train: np.ndarray = np.load(PATH_CIFAR10_AUGMENTED_TRAIN_IMAGES)
        self.y_train: np.ndarray = np.load(PATH_CIFAR10_AUGMENTED_TRAIN_LABELS)
        self.x_test: np.ndarray  = np.load(PATH_CIFAR10_TEST_IMAGES)
        self.y_test: np.ndarray  = np.load(PATH_CIFAR10_TEST_LABELS)
        
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(
                self.x_train.shape[0], 3, img_rows, img_cols
            )
            self.x_test = self.x_test.reshape(
                self.x_test.shape[0], 3, img_rows, img_cols
            )
            self.input_shape = (3, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(
                self.x_train.shape[0], img_rows, img_cols, 3
            )
            self.x_test = self.x_test.reshape(
                self.x_test.shape[0], img_rows, img_cols, 3
            )
            self.input_shape = (img_rows, img_cols, 3)
        
        self.x_train = self.x_train.astype('float32')
        self.x_train /= 255
        
        self.x_test  = self.x_test.astype('float32')
        self.x_test /= 255
        
        self.y_train = to_categorical(self.y_train, self.NUM_CLASSES)
        self.y_test  = to_categorical(self.y_test, self.NUM_CLASSES)
    
    def build_shallow_model(self, optimizer: str) -> None:
        """Sets a shallow model architecture with a given optimizer.
        
        The model has the following architecture:
            1 Conv2D layer with 32 filters, 3x3 kernel and relu activation.
            1 Conv2D layer with 64 filters, 3x3 kernel and relu activation.
            1 MaxPooling with a 2x2 size.
            Flatten the output of the Max Pooling.
            1 Dense layer with 128 outputs and relu activation.
            1 Dense layer with 10 outputs (1 per class) and softmax activation.
        
        Parameters
        ----------
        optimizer: str
            The algorithm used to reduce the loss function
        """
        
        self.model = Sequential()
        self.model.add(
            Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape)
        )
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=self.OPTIMIZERS[optimizer],
            metrics=['accuracy']
        )
    
    def build_deep_model(self, optimizer: str) -> None:
        """Sets a deep model architecture with a given optimizer.
        
        The model has the following architecture:
            1 Conv2D layer with 32 filters, 3x3 kernel and relu activation.
            4 Conv2D layer with 64 filters, 3x3 kernel and relu activation.
            1 MaxPooling with a 2x2 size.
            Flatten the output of the Max Pooling.
            2 Dense layer with 128 outputs and relu activation.
            1 Dense layer with 10 outputs (1 per class) and softmax activation.
        
        Parameters
        ----------
        optimizer: str
            The algorithm used to reduce the loss function
        """
        self.model = Sequential()
        self.model.add(
            Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape)
        )
        
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=self.OPTIMIZERS[optimizer],
            metrics=['accuracy']
        )
    
    def train_model(self) -> None:
        """The current model is trained and evaluated showing the training progress.
        """
        
        # Shows information about the network
        print(f'Model parameters = {self.model.count_params()}')
        print(self.model.summary())
        
        # The model is trained
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            verbose=1,
            validation_data=(self.x_test[:1000], self.y_test[:1000])
        )
        
        # The model is evaluated
        score = self.model.evaluate(self.x_test[1000:], self.y_test[1000:], verbose=0)
        
        # Shows the results of the evaluation
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    
    def save_model(self, path: str) -> None:
        """The current model is saved in the given path.
        
        Parameters
        ----------
        path: str
            The path where the model will be saved
        """
        
        self.model.save(path)