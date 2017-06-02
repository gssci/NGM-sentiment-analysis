import tensorflow as tf
import numpy as np
import os
from batch import batch_iter, test_batch_inter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
len_input = 1014
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 2, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def g(input_x,num_classes=2, filter_sizes=(7, 7, 3), frame_size=32, num_hidden_units=256,
      num_quantized_chars=70, dropout_keep_prob=0.5):

    a = tf.one_hot(
        indices=input_x,
        depth=70,
        axis=1,
        dtype=tf.float32)

    a = tf.expand_dims(a, 3)

    # Convolutional Layer 1
    with tf.name_scope("conv-maxpool-1"):
        filter_shape = [num_quantized_chars, filter_sizes[0], 1, frame_size]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
        conv = tf.nn.conv2d(a, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
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
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
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
    with tf.name_scope("conv-maxpool-3"):
        filter_shape = [1, filter_sizes[2], frame_size, frame_size]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
        conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 3, 1],
            padding='VALID',
            name="pool3")

    # Fully-connected Layer 1
    num_features_total = 36 * frame_size
    h_pool_flat = tf.reshape(pooled, [-1, num_features_total])

    with tf.name_scope("dropout-1"):
        drop1 = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    with tf.name_scope("fc-1"):
        W = tf.Variable(tf.truncated_normal([num_features_total, num_hidden_units], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
        fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")

    # Fully-connected Layer 2
    with tf.name_scope("dropout-2"):
        drop2 = tf.nn.dropout(fc_1_output, dropout_keep_prob)

    with tf.name_scope("fc-2"):
        W = tf.Variable(tf.truncated_normal([num_hidden_units, num_hidden_units], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
        fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

    # Fully-connected Layer 3
    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([num_hidden_units, num_classes], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")

    return scores


def train_neural_network():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            alpha1 = tf.constant(0.25, dtype=np.float32, name="a1")
            alpha2 = tf.constant(0.25, dtype=np.float32, name="a2")
            alpha3 = tf.constant(0.15, dtype=np.float32, name="a3")
            in_u1 = tf.placeholder(tf.int32, {None, len_input, }, name="ull")
            in_v1 = tf.placeholder(tf.int32, [None, len_input, ], name="vll")
            in_u2 = tf.placeholder(tf.int32, [None, len_input, ], name="ulu")
            in_v2 = tf.placeholder(tf.int32, [None, len_input, ], name="vlu")
            in_u3 = tf.placeholder(tf.int32, [None, len_input, ], name="ulu")
            in_v3 = tf.placeholder(tf.int32, [None, len_input, ], name="ulu")
            labels_u1 = tf.placeholder(tf.float32, [None, 2], name="lull")
            labels_v1 = tf.placeholder(tf.float32, [None, 2], name="lvll")
            labels_u2 = tf.placeholder(tf.float32, [None, 2], name="lulu")
            weights_ll = tf.placeholder(tf.float32, [None, ], name="wll")
            weights_lu = tf.placeholder(tf.float32, [None, ], name="wlu")
            weights_uu = tf.placeholder(tf.float32, [None, ], name="wuu")
            cu1 = tf.placeholder(tf.float32, [None, ], name="CuLL")
            cv1 = tf.placeholder(tf.float32, [None, ], name="CvLL")
            cu2 = tf.placeholder(tf.float32, [None, ], name="CuLU")

            scores_u1 = g(in_u1)
            scores_v1 = g(in_v1)
            scores_u2 = g(in_u2)

            loss_function = tf.reduce_mean(alpha1 * weights_ll * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u1, labels=scores_v1) \
                            + cu1 * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u1, labels=labels_u1) \
                            + cv1 * tf.nn.softmax_cross_entropy_with_logits(logits=scores_v1, labels=labels_v1)) \
                            + tf.reduce_mean(alpha2 * weights_lu * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u2, labels=g(in_v2)) \
                            + cu2 * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u2, labels=labels_u2)) \
                            + tf.reduce_mean(alpha3 * weights_uu * tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u3), labels=g(in_v3)))

            optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss_function, global_step=global_step)

            correct_predictions = tf.concat([tf.equal(tf.argmax(scores_u1, 1), tf.argmax(labels_u1, 1)),
                                             tf.equal(tf.argmax(scores_v1, 1), tf.argmax(labels_v1, 1)),
                                             tf.equal(tf.argmax(scores_u2, 1), tf.argmax(labels_u2, 1))],axis=0)
            train_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./summary')
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            batches = batch_iter(batch_size=128,num_epochs=20)

            for batch in batches:
                current_step = tf.train.global_step(sess, global_step)

                u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu = batch
                _, loss, acc, a1 = sess.run([optimizer, loss_function, train_accuracy, alpha1],
                                        feed_dict={in_u1: u1,
                                                   in_v1: v1,
                                                   in_u2: u2,
                                                   in_v2: v2,
                                                   in_u3: u3,
                                                   in_v3: v3,
                                                   labels_u1: lu1,
                                                   labels_v1: lv1,
                                                   labels_u2: lu2,
                                                   weights_ll: w_ll,
                                                   weights_lu: w_lu,
                                                   weights_uu: w_uu,
                                                   cu1: c_ull,
                                                   cv1: c_vll,
                                                   cu2: c_ulu})

                if current_step % FLAGS.evaluate_every == 0:
                    print("Step: " + str(current_step) + " Train Batch Acc.: " + str(acc) +
                          " Train Loss: " + str(loss) + " " + str(a1))

                if current_step % FLAGS.checkpoint_every == 0:
                    save_path = saver.save(sess, "./model.ckpt")
                    print("Model saved in file: %s" % save_path)
