from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from cnn import CNN
from mlp import MLP

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    indices = tf.random.shuffle(tf.range(len(train_inputs)))
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)

    for i in range(0, len(shuffled_inputs), model.batch_size):
        batch_inputs = shuffled_inputs[i:i + model.batch_size]
        batch_labels = shuffled_labels[i:i + model.batch_size]

        batch_inputs = tf.image.random_flip_left_right(batch_inputs)

        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #print(f"Batch {i // model.batch_size + 1}: Loss = {loss.numpy():.4f}")


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    # TODO: Implement the testing loop
    logits = model.call(test_inputs)
    accuracy = model.accuracy(logits, test_labels)
    return accuracy

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, logits, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def main():
    '''
    Read in CIFAR10 data (limited to a subset), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    Students should receive a final accuracy 
    on the testing examples for cat, deer and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = 'data/train'
    AUTOGRADER_TEST_FILE = 'data/test'

    # TODO: assignment.main() pt 1
    # Load your testing and training data using the get_data function

    # TODO: assignment.main() pt 2
    # Initialize your model and optimizer

    # TODO: assignment.main() pt 3
    # Train your model

    # TODO: assignment.main() pt 4
    # Test your model

    # TODO: assignment.main() pt 5
    # Save your predictions as either "predictions_cnn.npy" or "predictions_mlp.npy"
    #   depending on which model you are using
    # You will submit these prediction files to the autograder with predictions
    #    For the CAT, DEER, and DOG classes
    LOCAL_TRAIN_FILE = '/Users/komalb/Downloads/hw2-mlp-cnn-rohitharavindra08-main/data-2/train'
    LOCAL_TEST_FILE = '/Users/komalb/Downloads/hw2-mlp-cnn-rohitharavindra08-main/data-2/test'

    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, classes=[3, 5, 7])  # cat, deer, dog
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE, classes=[3, 5, 7])

    # Select model type: 'mlp' or 'cnn'
    model_type = 'mlp'  # Change this to 'mlp' if you want to train the MLP model
    if model_type == 'mlp':
        model = MLP(classes=[3, 5, 7])
    else:
        model = CNN(classes=[3, 5, 7])  # CNN model

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(10):
        train(model, optimizer, train_inputs, train_labels)
        logits_train = model.call(train_inputs)
        train_accuracy = model.accuracy(logits_train, train_labels)
        print(f'Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.4f}')

    logits_test = model.call(test_inputs, is_testing=True)
    test_accuracy = model.accuracy(logits_test, test_labels)
    print(f'Testing Accuracy: {test_accuracy:.4f}')

    predictions = tf.argmax(logits_test, axis=1).numpy()
    if model_type == 'mlp':
        np.save("predictions_mlp.npy", predictions)
    else:
        np.save("predictions_cnn.npy", predictions)


if __name__ == '__main__':
    main()
