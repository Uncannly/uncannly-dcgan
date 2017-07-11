import tensorflow as tf

from discriminator import discriminator


def setup_discriminators(initializer, Gz):
    real_in = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)  # Real images
    Dx = discriminator(real_in, initializer)  # Produces probabilities for real images
    Dg = discriminator(Gz, initializer, reuse=True)  # Produces probabilities for generator images

    # These functions together define the optimization objective of the GAN.
    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(tf.log(Dg))  # This optimizes the generator.

    tvars = tf.trainable_variables()

    # The below code is responsible for applying gradient descent to update the GAN.
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    d_grads = trainerD.compute_gradients(d_loss, tvars[9:])  # Only update the weights for the discriminator network.
    g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])  # Only update the weights for the generator network.

    update_D = trainerD.apply_gradients(d_grads)
    update_G = trainerG.apply_gradients(g_grads)

    return {
        'update_D': update_D,
        'update_G': update_G,
        'real_in': real_in,
        'd_loss': d_loss,
        'g_loss': g_loss
    }
