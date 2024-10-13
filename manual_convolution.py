from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None): 
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """

        #define some useful variables
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        # fill out the rest!
        assert input_in_channels == filter_in_channels, "Input channels and filter channels must match."

        # Padding logic for 'SAME' padding
        if self.padding == "SAME":
            pad_height = (filter_height - 1) // 2
            pad_width = (filter_width - 1) // 2
            inputs = tf.pad(inputs, [[0, 0], [pad_height, pad_height], [pad_width, pad_width], [0, 0]])

        # Output dimensions
        out_height = (in_height - filter_height + 2 * pad_height) // self.strides[1] + 1
        out_width = (in_width - filter_width + 2 * pad_width) // self.strides[2] + 1
        output = np.zeros((num_examples, out_height, out_width, filter_out_channels))

        # Perform manual convolution
        for example in range(num_examples):
            for h in range(out_height):
                for w in range(out_width):
                    for oc in range(filter_out_channels):
                        # Extract the region for convolution
                        h_start = h * self.strides[1]
                        h_end = h_start + filter_height
                        w_start = w * self.strides[2]
                        w_end = w_start + filter_width
                        region = inputs[example, h_start:h_end, w_start:w_end, :]

                        # Perform convolution
                        output[example, h, w, oc] = tf.reduce_sum(region * self.filters[:, :, :, oc])

        # Add bias if applicable
        if self.use_bias:
            output += self.bias

        # Convert the output to a Tensor
        return tf.convert_to_tensor(output, dtype=tf.float32)