import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def discriminator(bottom, initializer, reuse=False):
    dis1 = slim.convolution2d(bottom, 16, [4, 4], stride=[2, 2], padding="SAME", \
                              biases_initializer=None, activation_fn=lrelu, \
                              reuse=reuse, scope='d_conv1', weights_initializer=initializer)

    dis2 = slim.convolution2d(dis1, 32, [4, 4], stride=[2, 2], padding="SAME", \
                              normalizer_fn=slim.batch_norm, activation_fn=lrelu, \
                              reuse=reuse, scope='d_conv2', weights_initializer=initializer)

    dis3 = slim.convolution2d(dis2, 64, [4, 4], stride=[2, 2], padding="SAME", \
                              normalizer_fn=slim.batch_norm, activation_fn=lrelu, \
                              reuse=reuse, scope='d_conv3', weights_initializer=initializer)

    d_out = slim.fully_connected(slim.flatten(dis3), 1, activation_fn=tf.nn.sigmoid, \
                                 reuse=reuse, scope='d_out', weights_initializer=initializer)

    return d_out
