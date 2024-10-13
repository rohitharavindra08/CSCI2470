from abc import ABC, abstractmethod
import tensorflow as tf

class CifarModel(tf.keras.Model):
	@abstractmethod
	def call(self, inputs):
		pass

	def __init__(self):
		super(CifarModel, self).__init__()

	def compute_loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""
		# TODO: Implement the loss function
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		return tf.reduce_mean(loss)

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
		:return: the accuracy of the model as a Tensor
		"""
		# TODO: Implement the accuracy function
		predictions = tf.argmax(logits, axis=1)
		correct_labels = tf.argmax(labels, axis=1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, correct_labels), tf.float32))
		return accuracy