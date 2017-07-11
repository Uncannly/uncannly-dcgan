import tensorflow as tf
import tensorflow.contrib.slim as slim


def generator(z, initializer):

    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm, \
                              activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])

    gen1 = slim.convolution2d_transpose( \
        zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2], \
        padding="SAME",normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)

    gen2 = slim.convolution2d_transpose( \
        gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2], \
        padding="SAME",normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)

    gen3 = slim.convolution2d_transpose( \
        gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2], \
        padding="SAME",normalizer_fn=slim.batch_norm, \
        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)

    g_out = slim.convolution2d_transpose( \
        gen3,num_outputs=1,kernel_size=[32,32],padding="SAME", \
        biases_initializer=None,activation_fn=tf.nn.tanh, \
        scope='g_out', weights_initializer=initializer)

    return g_out
