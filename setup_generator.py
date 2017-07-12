import tensorflow as tf

from constants import Z_SIZE
from generator import generator


def setup_generator():
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    z_in = tf.placeholder(shape=[None, Z_SIZE], dtype=tf.float32)
    Gz = generator(z_in, initializer)

    return {
        'Gz': Gz,
        'z_in': z_in,
        'initializer': initializer
    }
