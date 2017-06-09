import tensorflow as tf
import numpy as np
import os
from create_input_encodings import string_to_int8_conversion
from neural_graph_machine import g

len_input = 1014
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        in_u1 = tf.placeholder(tf.int32, {None, len_input, }, name="ull")

        with tf.variable_scope("ngm") as scope:
            scores_u1 = g(in_u1, dropout_keep_prob=1.0)
            def evaluate_user_input(in_x):
                scores_x = sess.run(tf.nn.softmax(scores_u1), feed_dict={
                    in_u1: in_x
                })
                return scores_x

        saver = tf.train.Saver()

        saver.restore(sess, "./model/model.ckpt")

        while(True):
            input_string = input("Enter input to evaluate: ")
            encoded_input = np.zeros(shape=(1,len_input), dtype=np.int32)
            encoded_input[0] = string_to_int8_conversion(input_string)
            result = evaluate_user_input(encoded_input)[0]

            s = ['Negative', 'Positive']

            print(s[np.argmax(result)])

            print("P(Positive) = " + str(result[1]))
            print("P(Negative) = " + str(result[0]))
