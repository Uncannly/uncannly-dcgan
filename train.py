import numpy as np
import os
import tensorflow as tf

import input_data
from constants import SAMPLE_DIRECTORY, MODEL_DIRECTORY, Z_SIZE, BATCH_SIZE
from save_images import save_images
from setup_discriminators import setup_discriminators
from setup_generator import setup_generator

mnist = input_data.read_data_sets("MNIST_data/")
tf.reset_default_graph()

generator_stuff = setup_generator()
z_in = generator_stuff['z_in']
initializer = generator_stuff['initializer']
Gz = generator_stuff['Gz']

discriminator_stuff = setup_discriminators(initializer, Gz)
update_D = discriminator_stuff['update_D']
update_G = discriminator_stuff['update_G']
real_in = discriminator_stuff['real_in']
d_loss = discriminator_stuff['d_loss']
g_loss = discriminator_stuff['g_loss']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500000):
        zs = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, Z_SIZE]).astype(np.float32)  # Generate a random z batch
        xs, _ = mnist.next_batch(BATCH_SIZE)  # Draw a sample batch from MNIST dataset.
        xs = (np.reshape(xs, [BATCH_SIZE, 28, 28, 1]) - 0.5) * 2.0  # Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                        constant_values=(-1, -1))  # Pad the images so the are 32x32
        _, dLoss = sess.run([update_D, d_loss], feed_dict={z_in: zs, real_in: xs})  # Update the discriminator
        _, gLoss = sess.run([update_G, g_loss], feed_dict={z_in: zs})  # Update the generator, twice for good measure.
        _, gLoss = sess.run([update_G, g_loss], feed_dict={z_in: zs})
        if i % 10 == 0:
            print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
            z2 = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, Z_SIZE]).astype(np.float32)  # Generate another z batch
            newZ = sess.run(Gz, feed_dict={z_in: z2})  # Use new z to get sample images from generator.
            if not os.path.exists(SAMPLE_DIRECTORY):
                os.makedirs(SAMPLE_DIRECTORY)
            # Save sample generator images for viewing training progress.
            save_images(np.reshape(newZ[0:36], [36, 32, 32]), [6, 6], SAMPLE_DIRECTORY + '/fig' + str(i) + '.png')
        if i % 100 == 0 and i != 0:
            if not os.path.exists(MODEL_DIRECTORY):
                os.makedirs(MODEL_DIRECTORY)
            tf.train.Saver().save(sess, MODEL_DIRECTORY + '/model-' + str(i) + '.cptk')
            print "Saved Model"
