import tensorflow as tf

from generator import generator
from src.constants import DATA_SIZE, Z_SIZE, THIS_MAGIC_NUMBER
from xavier_init import xavier_init


def setup_generator():
    z_in = tf.placeholder(tf.float32, shape=[None, Z_SIZE], name='z_in')

    G_W1 = tf.Variable(xavier_init([Z_SIZE, THIS_MAGIC_NUMBER]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[THIS_MAGIC_NUMBER]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([THIS_MAGIC_NUMBER, DATA_SIZE]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[DATA_SIZE]), name='G_b2')

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    return {
        'Gz': generator(z_in, G_W1, G_W2, G_b1, G_b2),
        'theta_G': theta_G,
        'z_in': z_in,
    }
