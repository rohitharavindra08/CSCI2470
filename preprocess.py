import pickle
import numpy as np
import tensorflow as tf
import os


def unpickle(file) -> dict[str, np.ndarray]:
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.
    NOTE: DO NOT EDIT
    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_next_batch(idx, inputs, labels, batch_size=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a numpy matrix of size 5 * 32 * 32 * 3, and the labels returned will be a numpy matrix of size 5 * 10.
    """
    return (inputs[idx*batch_size:(idx+1)*batch_size], np.array(labels[idx*batch_size:(idx+1)*batch_size]))


def get_data(file_path, classes) -> tuple[np.ndarray, tf.Tensor]:
    """
    Given a file path and a list of class indices, returns an array of 
    normalized inputs (images) and an array of labels. 
    
    - **Note** that because you are using tf.one_hot() for your labels, your
    labels will be a Tensor, hence the mixed output typing for this function. This 
    is fine because TensorFlow also works with NumPy arrays, which you will
    see more of in the next assignment. 

    :param file_path: file path for inputs and labels, something 
                        like 'CIFAR_data_compressed/train'
    :param classes: list of class labels (0-9) to include in the dataset

    :return: normalized NumPy array of inputs and tensor of labels, where 
                inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and 
                Tensor of labels with size (num_examples, num_classes)
    """
    unpickled_file: dict[str, np.ndarry] = unpickle(file_path)
    inputs: np.ndarry = np.array(unpickled_file[b'data'])
    labels: np.ndarry = np.array(unpickled_file[b'labels'])

    # TODO: Extract only the data that matches the corresponding classes we want
    filtered_indices = np.isin(labels, classes)
    inputs = inputs[filtered_indices]
    labels = labels[filtered_indices]

    # Reshape the inputs to (num_examples, 32, 32, 3)
    inputs = inputs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Normalize the inputs to [0, 1]
    inputs = inputs.astype(np.float32) / 255.0

    # Map labels to new indices (e.g., 0 for "cat", 1 for "deer", 2 for "dog")
    label_mapping = {classes[i]: i for i in range(len(classes))}
    mapped_labels = np.array([label_mapping[label] for label in labels])

    # Convert labels to one-hot vectors
    one_hot_labels = tf.one_hot(mapped_labels, len(classes))

    return inputs, one_hot_labels