from __future__ import absolute_import
import tensorflow as tf

class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int] = [1, 1, 1, 1], padding="VALID", use_bias=True, *args, **kwargs):
        """
        Custom Convolution Layer.
        :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
        :param strides: Strides for the convolution. Default is [1, 1, 1, 1].
        :param padding: Either 'SAME' or 'VALID'.
        :param use_bias: Whether to use bias in the convolution.
        """
        super(ManualConv2d, self).__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=0.05), name=name, trainable=trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable=True)
        self.use_bias = use_bias
        if use_bias:
            self.bias = get_var("conv_bias", [filter_shape[-1]], trainable=True)
        else:
            self.bias = None

    def get_weights(self):
        if self.bias is not None:
            return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None):
        self.filters = filters
        if bias is not None:
            self.bias = bias

    def call(self, inputs):
        """
        Custom forward pass for the convolution layer.
        :param inputs: Input tensor with shape [num_examples, in_height, in_width, in_channels]
        """
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        assert input_in_channels == filter_in_channels, "Input channels and filter channels must match."

        # Padding logic
        if self.padding == "SAME":
            pad_height = max((in_height - 1) * self.strides[1] + filter_height - in_height, 0)
            pad_width = max((in_width - 1) * self.strides[2] + filter_width - in_width, 0)
            pad_height_top = pad_height // 2
            pad_height_bottom = pad_height - pad_height_top
            pad_width_left = pad_width // 2
            pad_width_right = pad_width - pad_width_left
            inputs = tf.pad(inputs, [[0, 0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
            out_height = (in_height + pad_height - filter_height) // self.strides[1] + 1
            out_width = (in_width + pad_width - filter_width) // self.strides[2] + 1
        elif self.padding == "VALID":
            out_height = (in_height - filter_height) // self.strides[1] + 1
            out_width = (in_width - filter_width) // self.strides[2] + 1

        # TensorFlow-friendly convolution computation
        output = tf.TensorArray(tf.float32, size=num_examples)
        for example in range(num_examples):
            example_output = tf.TensorArray(tf.float32, size=out_height)
            for h in range(out_height):
                row_output = tf.TensorArray(tf.float32, size=out_width)
                for w in range(out_width):
                    conv_region = inputs[example, h * self.strides[1]:h * self.strides[1] + filter_height,
                                         w * self.strides[2]:w * self.strides[2] + filter_width, :]
                    conv_value = [tf.reduce_sum(conv_region * self.filters[:, :, :, oc]) for oc in range(filter_out_channels)]
                    row_output = row_output.write(w, conv_value)
                example_output = example_output.write(h, row_output.stack())
            output = output.write(example, example_output.stack())

        output = output.stack()

        # Add bias if applicable
        if self.use_bias:
            output += self.bias

        return output

# The wrapper function 'stu_conv2d'
def stu_conv2d(inputs, filters, strides=[1, 1, 1, 1], padding="VALID", use_bias=True):
    """
    Wrapper function to perform a 2D convolution using ManualConv2d.
    :param inputs: Input tensor (e.g., [num_examples, in_height, in_width, in_channels])
    :param filters: Filter tensor (e.g., [filter_height, filter_width, in_channels, out_channels])
    :param strides: Stride of the convolution. Default is [1, 1, 1, 1].
    :param padding: Either "VALID" or "SAME".
    :param use_bias: Whether to use bias in the convolution layer.
    :return: The result of the convolution.
    """
    # Create an instance of the ManualConv2d layer
    conv_layer = ManualConv2d(filter_shape=filters.shape, strides=strides, padding=padding, use_bias=use_bias)
    
    # Set the filter weights and bias for the layer
    conv_layer.set_weights(filters)
    
    # Call the layer to perform the convolution
    return conv_layer(inputs)