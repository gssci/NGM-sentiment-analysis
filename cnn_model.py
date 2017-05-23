# Based on implementation by Sadegh Charmchi: https://github.com/scharmchi/char-level-cnn-tf
import tensorflow as tf


class CNNModel(object):

    def __init__(self, alpha1=0.2, alpha2=0.4, num_classes=2, filter_sizes=(7, 7, 3), frame_size=32, num_hidden_units=256,
                 sequence_max_length=1014, num_quantized_chars=70):

        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Convolutional Layer 1
        with tf.name_scope("conv-maxpool-1"):
            filter_shape = [num_quantized_chars, filter_sizes[0], 1, frame_size]
            W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1,1,3,1],
                strides=[1,1,3,1],
                padding='VALID',
                name="pool1")

        # Convolutional Layer 2
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [1, filter_sizes[1], frame_size, frame_size]
            W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool2")

        # Convolutional Layer 3
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [1, filter_sizes[5], frame_size, frame_size]
            W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='VALID',
                name="pool3")

        # Fully-connected Layer 1
        num_features_total = 34 * frame_size
        h_pool_flat = tf.reshape(pooled, [-1, num_features_total])

        with tf.name_scope("dropout-1"):
            drop1 = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("fc-1"):
            W = tf.Variable(tf.random_normal([num_features_total, num_hidden_units], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")

        # Fully-connected Layer 2
        with tf.name_scope("dropout-2"):
            drop2 = tf.nn.dropout(fc_1_output, self.dropout_keep_prob)

        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.random_normal([num_hidden_units, num_hidden_units], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

        # Fully-connected Layer 3
        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.random_normal([num_hidden_units, num_classes], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            predictions = tf.argmax(scores, 1, name="predictions")