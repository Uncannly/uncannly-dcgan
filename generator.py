import tensorflow as tf


def generator(z_in, G_W1, G_W2, G_b1, G_b2):
    G_h1 = tf.nn.relu(tf.matmul(z_in, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob
