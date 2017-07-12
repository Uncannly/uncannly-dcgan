import tensorflow as tf

from constants import DATA_SIZE_SQRT
from discriminator import discriminator


def setup_discriminators(initializer, Gz):
    real_in = tf.placeholder(shape=[None, DATA_SIZE_SQRT, DATA_SIZE_SQRT, 1], dtype=tf.float32)
    Dx = discriminator(real_in, initializer)
    Dg = discriminator(Gz, initializer, reuse=True)

    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))
    g_loss = -tf.reduce_mean(tf.log(Dg))

    tvars = tf.trainable_variables()

    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    d_grads = trainerD.compute_gradients(d_loss, tvars[9:])
    g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])

    update_D = trainerD.apply_gradients(d_grads)
    update_G = trainerG.apply_gradients(g_grads)

    return {
        'update_D': update_D,
        'update_G': update_G,
        'real_in': real_in,
        'd_loss': d_loss,
        'g_loss': g_loss
    }
