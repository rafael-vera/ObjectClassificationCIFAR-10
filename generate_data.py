import os

import numpy as np
from PIL import Image
import keras
from keras.datasets import cifar10 # Here is the dataset

from file_paths import *

def generate_data(with_augment: bool = False) -> None:
    """Generates and saves data by dividing it into training data and test data,
    optionally data augmentation can be applied
    
    Parameters
    ----------
    with_augment: bool, optional
        Indicates whether the data will be augmented (default is False)
    """
    
    # The CIFAR-10 dataset is loaded
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # The train data are randomly arranged
    idx = np.argsort(
        np.random.random(
            y_train.shape[0]
        )
    )
    x_train = x_train[idx]
    y_train = y_train[idx]
    
    # The test data are randomly arranged
    idx = np.argsort(
        np.random.random(
            y_test.shape[0]
        )
    )
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    # If a file don't exist, it is created
    if not os.path.exists(PATH_CIFAR10_TRAIN_IMAGES):
        np.save(PATH_CIFAR10_TRAIN_IMAGES, x_train)
    if not os.path.exists(PATH_CIFAR10_TRAIN_LABELS):
        np.save(PATH_CIFAR10_TRAIN_LABELS, y_train)
    if not os.path.exists(PATH_CIFAR10_TEST_IMAGES):
        np.save(PATH_CIFAR10_TEST_IMAGES, x_test)
    if not os.path.exists(PATH_CIFAR10_TEST_LABELS):
        np.save(PATH_CIFAR10_TEST_LABELS, y_test)
    
    # If the with_augment variable is True,
    # then it starts the data augmentation process
    if with_augment:
        factor = 10
        dim = 28
        z = (32 - dim) / 2
        new_x_train = np.zeros((x_train.shape[0]*factor, dim, dim, 3), dtype='uint8')
        new_y_train = np.zeros(y_train.shape[0]*factor, dtype='uint8')
        k: int = 0
        
        for i in range(x_train.shape[0]):
            img = Image.fromarray(x_train[i,:])
            img = img.crop((z, z, 32-z, 32-z))
            new_x_train[k, ...] = np.array(img)
            new_y_train[k] = y_train[i]
            k += 1
            
            for j in range(factor-1):
                new_x_train[k, ...] = augment(x_train[i,:], dim)
                new_y_train[k] = y_train[i]
                k += 1
        
        idx = np.argsort(np.random.random(new_x_train.shape[0]))
        new_x_train = new_x_train[idx]
        new_y_train = new_y_train[idx]
        
        np.save(PATH_CIFAR10_AUGMENTED_TRAIN_IMAGES, new_x_train)
        np.save(PATH_CIFAR10_AUGMENTED_TRAIN_LABELS, new_y_train)
        
        new_x_test = np.zeros((x_test.shape[0], dim, dim, 3), dtype='uint8')
        for i in range(x_test.shape[0]):
            img = Image.fromarray(x_test[i,:])
            img = img.crop((z, z, 32-z, 32-z))
            new_x_test[i, ...] = np.array(img)
        np.save(PATH_CIFAR10_AUGMENTED_TEST_IMAGES, new_x_test)


def augment(img: np.ndarray, dim: int) -> np.ndarray:
    """Modify the image with 50% probability of flipping the image
    and 33.33% probability of rotating the image
    
    Parameters
    ----------
    img: Ndarray
        The image represented as an Numpy array
    dim: int
        The new dimension of the image
    
    Returns
    -------
    Ndarray
        The new image represented as an Numpy array
    """
    
    image = Image.fromarray(img)
    
    if np.random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() < 0.3333:
        z = (32 - dim) / 2
        r = 10 * np.random.random() - 5
        image = image.rotate(r, resample=Image.BILINEAR)
        image = image.crop((z, z, 32-z, 32-z))
    else:
        x = int((32-dim-1) * np.random.random())
        y = int((32-dim-1) * np.random.random())
        image = image.crop((x, y, x+dim, y+dim))
    return np.array(image)