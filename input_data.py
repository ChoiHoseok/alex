import numpy as np
import pickle
import os

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file
def one_hot_encoded(class_numbers, num_classes=None):
    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]
def _get_file_path(filename=""):
    data_path = './'
    return os.path.join(data_path, "cifar-10-batches-py/", filename)
def _unpickle(filename):
    file_path = _get_file_path(filename)
    if not (filename=='test_batch'):
        print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data
def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    return images
def _load_data(filename):
    data = _unpickle(filename)
    # Get the raw images.
    raw_images = data[b'data']
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    # Convert the images.
    images = _convert_images(raw_images)
    return images, cls
def load_training_data():
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images_train], dtype=int)

    begin = 0

    for i in range(num_files_train):
        images_batch, cls_batch = _load_data(filename='data_batch_'+str(i+1))
        num_images = len(images_batch)
        end = begin + num_images

        images[begin:end,:] = images_batch

        cls[begin:end] = cls_batch

        begin = end
    return images, cls, one_hot_encoded(class_numbers=cls,num_classes = num_classes)
def load_test_data():
    images, cls = _load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


load_training_data()
