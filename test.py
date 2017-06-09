import tensorflow as tf
import os
import numpy as np
from batch import test_batch_inter
from neural_graph_machine import g
len_input = 1014
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        in_u1 = tf.placeholder(tf.int32, {None, len_input, }, name="ull")
        labels_u1 = tf.placeholder(tf.float32, [None, 2], name="lull")

        with tf.variable_scope("ngm") as scope:
            scores_u1 = g(in_u1, dropout_keep_prob=1.0)

        correct_predictions = tf.concat(tf.equal(tf.argmax(scores_u1, 1), tf.argmax(labels_u1, 1)),axis=0)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        accs = list()

        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")

        batches = test_batch_inter(batch_size=250)

        for batch in batches:
            acc = sess.run(accuracy, feed_dict={in_u1:batch[0], labels_u1:batch[1]})
            accs.append(acc)
            print("Last Batch Acc: " + str(acc))

        print("Total Accuracy: " + str(np.mean(accs)))