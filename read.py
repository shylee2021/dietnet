from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def read_something_blahblah(filename):
    filename_q = tf.train.string_input_producer([filename],
                                                num_epoch=None)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_q)

    features = tf.parse_single_example(
        value,
        features={
            'label': tf.FixedLenFeature([], tf.int32)
            'image': tf.FixedLenFeature([dim], tf.int32)
        })