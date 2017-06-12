from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from dietnet import DietNet

def train(args):
    queue = tf.FIFOQueue(capacity=50, dtype=[tf.float32, tf.int32], shape=[[4, []]])

    enq_op = queue.enqueue_many([data, labels])
    data_batch, labels_batch = queue.dequeue()

    q_runner = tf.train.QueueRunner(queue, [enq_op] * NUM_THREADS)
    with tf.Session() as sess:
        # create coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        enq_threads = q_runner.create_threads(sess, coord=coord, start=True)

        for step in xrange(100):
            if coord.should_stop():
                break
            data_samples, label_samples = sess.run([data_batch, labels_batch])
        coord.request_stop()
        coord.join(enq_threads)

