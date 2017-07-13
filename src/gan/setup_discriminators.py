import tensorflow as tf

from discriminator import discriminator
from xavier_init import xavier_init


def setup_discriminators(Gz):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

    D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    D_real, D_logit_real = discriminator(X, D_W1, D_W2, D_b1, D_b2)
    D_fake, D_logit_fake = discriminator(Gz, D_W1, D_W2, D_b1, D_b2)

    d_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    g_loss = -tf.reduce_mean(tf.log(D_fake))

    return {
        'theta_D': theta_D,
        'd_loss': d_loss,
        'g_loss': g_loss,
        'X': X,
    }
