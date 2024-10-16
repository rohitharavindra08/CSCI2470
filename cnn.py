from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.hidden_layer_size = 320

        self.epsilon = 1e-3  # this is used for batch normalization only!

        # CNN architecture for training
        # 1st Convolutional Layer
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # 2nd Convolutional Layer
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # 3rd Convolutional Layer
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

        # Custom ManualConv2d layer for testing
        self.manual_conv = ManualConv2d([3, 3, 3, 32], padding="SAME")

        # Flatten Layer
        self.flatten = tf.keras.layers.Flatten()

        # Dense Layers
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.output_layer = tf.keras.layers.Dense(self.num_classes)  # Final layer (logits), no activation

    def call(self, inputs, is_testing=False):
        if is_testing:
            # Use the custom ManualConv2d layer for testing
            x = self.manual_conv(inputs)
        else:
            # Use TensorFlow's Conv2D layers for training
            x = self.conv1(inputs)
            x = self.batch_norm1(x)
            x = tf.nn.relu(x)
            x = self.pool1(x)

        # Continue with the rest of the CNN architecture (for both training and testing)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.dropout2(x)

        logits = self.output_layer(x)

        return logits
