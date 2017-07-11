import tensorflow as tf

from constants import Z_SIZE
from generator import generator


def setup_generator():
    # This initializer is used to initialize all the weights of the network.
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    z_in = tf.placeholder(shape=[None, Z_SIZE], dtype=tf.float32)  # Random vector
    Gz = generator(z_in, initializer)  # Generates images from random z vectors

    return {
        'Gz': Gz,
        'z_in': z_in,
        'initializer': initializer
    }
