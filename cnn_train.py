import tensorflow as tf
import numpy as np
import cnn_preprocessing
from cnn_model import CharCNN
import os
from embeddings_graph import EmbeddingsGraph
from cnn_preprocessing import encode_review

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


g = EmbeddingsGraph()
# to find some coherence in this elegant nonsense, I need to create different batch sizes
# for each term of the objective function, otherwise in an epoch
# I would have finished the labeled samples before computing
# the distances of the connected samples
n_epochs = 20
batch_size = 128


edges = g.edges
start = 0
finish = 128
edges_weights = g.edges_weights
n_batches = int(len(edges) / batch_size)+1

def label(i):
    if 0 <= i < 12500:
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def next_batch():
    """
    TODO: return batch of labeled instances, their labels, their neighbors, and the weights 
    :return: 
    """
    global start
    global finish
    edges_ll = []
    edges_lu = []
    edges_uu = []
    w_ll = []
    w_lu = []
    w_uu = []
    edg = edges[start:finish]

    for (i,j) in edg:
        if (0 <= i < 25000) and (0 <= j < 25000):
            edges_ll.append((i, j))
            w_ll.append(edges_weights.get((i,j)))
        elif (0 <= i < 25000) and (25000 <= j < 75000):
            edges_lu.append((i, j))
            w_lu.append(edges_weights.get((i,j)))
        else:
            edges_uu.append((i, j))
            w_uu.append(edges_weights.get((i,j)))

    u1 = np.vstack([encode_review(u) for u,v in edges_ll])
    lu1 = np.vstack([label(u) for u,v in edges_ll])
    v1 = np.vstack([encode_review(v) for u, v in edges_ll])
    lv1 = np.vstack([label(v) for u, v in edges_ll])

    u2 = np.vstack([encode_review(u) for u,v in edges_lu])
    lu2 = np.vstack([label(u) for u,v in edges_lu])
    v2 = np.vstack([encode_review(v) for u, v in edges_lu])

    u3 = np.vstack([encode_review(u) for u,v in edges_uu])
    v3 = np.vstack([encode_review(v) for u, v in edges_uu])

    start = finish
    finish += 128

    if finish>len(edges):
        finish = len(edges)

    return u1, v1, lu1, lv1, u2, v2, lu2, u3, v3, w_ll, w_lu, w_uu


def g(input_x,num_classes=2, filter_sizes=(7, 7, 3), frame_size=32, num_hidden_units=256,
      num_quantized_chars=70, dropout_keep_prob=0.5):

    # Convolutional Layer 1
    with tf.name_scope("conv-maxpool-1"):
        filter_shape = [num_quantized_chars, filter_sizes[0], 1, frame_size]
        W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
        conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
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
    with tf.name_scope("conv-maxpool-3"):
        filter_shape = [1, filter_sizes[2], frame_size, frame_size]
        W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
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
        W = tf.Variable(tf.random_normal([num_features_total, num_hidden_units], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
        fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")

    # Fully-connected Layer 2
    with tf.name_scope("dropout-2"):
        drop2 = tf.nn.dropout(fc_1_output, dropout_keep_prob)

    with tf.name_scope("fc-2"):
        W = tf.Variable(tf.random_normal([num_hidden_units, num_hidden_units], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden_units]), name="b")
        fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

    # Fully-connected Layer 3
    with tf.name_scope("output"):
        W = tf.Variable(tf.random_normal([num_hidden_units, num_classes], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
        predictions = tf.argmax(scores, 1, name="predictions")

    return scores

def train_neural_network():
    """
    :param x: index of sample
    :return: 
    """
    global edges

    alpha1 = tf.constant(0.1, dtype=np.float32, name="a1")
    alpha2 = tf.constant(0.4, dtype=np.float32, name="a2")
    alpha3 = tf.constant(0.4, dtype=np.float32, name="a2")
    in_u1 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="ull")
    in_v1 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="vll")
    in_u2 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="ulu")
    in_v2 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="vlu")
    in_u3 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="ulu")
    in_v3 = tf.placeholder(tf.float32, [None, 70, 1014, 1], name="ulu")
    labels_u1 = tf.placeholder(tf.float32, [None, 2], name="lull")
    labels_v1 = tf.placeholder(tf.float32, [None, 2], name="lvll")
    labels_u2 = tf.placeholder(tf.float32, [None, 2], name="lulu")
    weights_ll = tf.placeholder(tf.float32, [None,], name="neighbors_weights")
    weights_lu = tf.placeholder(tf.float32, [None,], name="neighbors_weights")
    weights_uu = tf.placeholder(tf.float32, [None,], name="neighbors_weights")

    loss_function = tf.reduce_sum(alpha1 * weights_ll * tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u1),labels=g(in_v1))) \
                    + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u1),labels=labels_u1)) \
                    + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=g(in_v1),labels=labels_v1)) \
                    + tf.reduce_sum(alpha2 * weights_lu * tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u2),labels=g(in_v2))) \
                    + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u2),labels=labels_u2)) \
                    + tf.reduce_sum(alpha3 * weights_uu * tf.nn.softmax_cross_entropy_with_logits(logits=g(in_u3),labels=g(in_v3)))

    loss = tf.reduce_mean(loss_function)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    n_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            edges = np.random.permutation(edges)
            epoch_loss = 0
            for _ in range(n_batches):
                u1, v1, lu1, lv1, u2, v2, lu2, u3, v3, w_ll, w_lu, w_uu = next_batch()
                _, c = sess.run([optimizer, loss],
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
                                           weights_uu: w_uu})
                epoch_loss += c
                print(str(epoch_loss))

            print('Epoch ', epoch+1, 'completed out of ', n_epochs, ' loss:', epoch_loss)
