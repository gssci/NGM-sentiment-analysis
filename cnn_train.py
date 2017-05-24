import tensorflow as tf
import numpy as np
import cnn_preprocessing
from cnn_model import CharCNN
import os
from embeddings_graph import EmbeddingsGraph
from cnn_preprocessing import encode_review

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

num_classes=2
filter_sizes=(7, 7, 3)
frame_size=32
num_hidden_units=256
sequence_max_length=1014
num_quantized_chars=70
num_neighbors=10

input = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")
labels = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="labels")
neighbors_l = tf.placeholder(tf.int16, [None, None, num_quantized_chars, sequence_max_length, 1], name="neighbors_x")
neighbors_u = tf.placeholder(tf.int16, [None, None, num_quantized_chars, sequence_max_length, 1], name="neighbors_x")
weights_ll = tf.placeholder(tf.float32, [None, None], name="neighbors_weights")
weights_lu = tf.placeholder(tf.float32, [None, None], name="neighbors_weights")
weights_uu = tf.placeholder(tf.float32, [None, None], name="neighbors_weights")

def cnn(input):
    # Convolutional Layer 1
    with tf.name_scope("conv-maxpool-1"):
        filter_shape = [num_quantized_chars, filter_sizes[0], 1, frame_size]
        W = tf.Variable(tf.random_normal(filter_shape, stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[frame_size]), name="b")
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 3, 1],
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
        filter_shape = [1, filter_sizes[2], frame_size, frame_size]
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



g = EmbeddingsGraph()


# to find some coherence in this elegant nonsense, I need to create different batch sizes
# for each term of the objective function, otherwise in an epoch
# I would have finished the labeled samples before computing
# the distances of the connected samples
n_epochs = 20
batch_size = 128
num_samples = 25000
n_batches = int(num_samples / batch_size)+1
batch_size_ll = int
batch_size_lu =
batch_size_ll =

# use alpha terms to correct disparities in batch sizes (?)
#
alpha1 = tf.constant(0.1,dtype=np.float32, name="a1")
alpha2 = tf.constant(0.4,dtype=np.float32, name="a2")
alpha3 = tf.constant(0.4,dtype=np.float32, name="a2")

def next_batch():
    """
    TODO: return batch of labeled instances, their labels, their neighbors, and the weights 
    :return: 
    """
    global start

    # shuffle at each epoch
    shuffle_indices = np.random.permutation(np.arange(num_samples))
    start = 0

    return

def train_neural_network(x):
    """
    :param x: index of sample
    :return: 
    """
    score = cnn(x)
    scores_neigh_l = cnn(neighbors_l)
    scores_neigh_u = cnn(neighbors_u)
    losses = tf.nn.softmax_cross_entropy_with_logits(score,labels) +\
             alpha1 * weights_ll * tf.nn.softmax_cross_entropy_with_logits()
    cost = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range():
                epoch_x, epoch_y, neighbors_ll, neighbors_lu, weights_ll, weights_lu = next_batch()
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, labels: epoch_y, neighbors_l: neighbors_ll, neighbors_u: neighbors_lu, weights_l: weights_ll, weights_u: weights_lu, dropout_keep_prob: 0.5})
                epoch_loss += c

            print('Epoch ', epoch+1, 'completed out of ', n_epochs, ' loss:', epoch_loss)

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))