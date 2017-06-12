from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class DietNet(object):
    """docstring for DietNet"""
    def __init__(self, inputs, targets, input_size, hidden_size, num_classes,
                 gamma=1, is_training=False, reuse=False):
        with tf.variable_scope("DietNet"):
            # basic inputs
            self.inputs = inputs
            self.targets = targets
            self.inputs_t = tf.transpose(self.inputs)

            # creating parameter matrix for hidden layer
            auxnet_embed = AuxNet(self.inputs_t, hidden_size, reuse=reuse)
            self.W_e = auxnet_embed.outputs

            # hidden layer
            self.hidden_pre = tf.matmul(self.inputs, self.W_e)
            bias1 = tf.get_variable(
                name="bias1", shape=[hidden_size], initializer=tf.zeros_initializer)
            self.hidden_pre = tf.add(self.hidden_pre, bias1, name="hidden_pre")
            self.hidden = tf.nn.relu(self.hidden_pre, name="hidden")
            self.hidden = tf.layers.dropout(
                self.hidden, name="dropped hidden")

            # prediction network
            self.logits = tf.layers.dense(
                inputs=self.hidden, units=num_classes,
                kernel_initializer=None, name="logits")

            # creating parameter matrix for reconstruction layer
            auxnet_recon = AuxNet(self.inputs_t, hidden_size, reuse=reuse)
            self.W_d = auxnet_recon.outputs
            self.W_d_t = tf.transpose(self.W_d)

            # reconstruction network
            self.inputs_recon = tf.matmul(self.hidden, self.W_d_t)
            bias2 = tf.get_variable(
                name="bias2", shape=[input_size], initializer=tf.zeros_initializer)
            self.inputs_recon = tf.add(self.inputs_recon, bias2, name="reconstruction")

            # losses
            self.class_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.targets, logits=self.logits, name="class_loss")

            self.recon_loss = tf.nn.l2_loss(
                self.inputs-self.inputs_recon, name="recon_loss")

            self.total_loss = tf.add(
                self.class_loss, gamma * self.recon_loss, name="total loss")


def embed(inputs):
    pass


class AuxNet(object):
    def __init__(self, inputs, hidden_size, out_size,
                 drop_prob=0.5, is_training=False, reuse=None):
        with tf.variable_scope("AuxNet"):
            self.inputs = inputs
            self.hidden = tf.layers.dense(
                inputs=self.inputs, units=hidden_size, activation=tf.nn.relu,
                kernel_initializer=None, name="hidden", reuse=reuse)
            self.dropped = tf.layers.dropout(
                inputs=self.hidden, rate=drop_prob,
                training=is_training, name="dropped hidden")
            self.outputs = tf.layers.dense(
                inputs=self.dropped, units=out_size,
                kernel_initializer=None, name="outputs", reuse=reuse)
