import tensorflow as tf

from generator import generator
from xavier_init import xavier_init


def setup_generator():
    z_in = tf.placeholder(tf.float32, shape=[None, 100], name='z_in')

    G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    return {
        'Gz': generator(z_in, G_W1, G_W2, G_b1, G_b2),
        'theta_G': theta_G,
        'z_in': z_in,
    }