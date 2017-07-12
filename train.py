import numpy as np
import os
import tensorflow as tf

import input_data
from constants import MODEL_DIRECTORY, BATCH_SIZE, DATA_SIZE_SQRT
from generate_random_z_batch import generate_random_z_batch
from output_words import output_words
from setup_discriminators import setup_discriminators
from setup_generator import setup_generator

data_sets = input_data.read_data_sets()
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

NUM_ITERATIONS = 500000
OUTPUT_WORDS_EVERY_N_ITERATIONS = 10
SAVE_MODEL_EVERY_N_ITERATIONS = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_ITERATIONS):

        zs = generate_random_z_batch()
        xs = data_sets.next_batch()
        xs = (np.reshape(xs, [BATCH_SIZE, DATA_SIZE_SQRT, DATA_SIZE_SQRT, 1]) - 0.5) * 2.0
        _, dLoss = sess.run([update_D, d_loss], feed_dict={z_in: zs, real_in: xs})
        _, gLoss = sess.run([update_G, g_loss], feed_dict={z_in: zs})
        _, gLoss = sess.run([update_G, g_loss], feed_dict={z_in: zs})

        if i % OUTPUT_WORDS_EVERY_N_ITERATIONS == 0:
            print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
            output_words(sess, generator_stuff, str(i))

        if i % SAVE_MODEL_EVERY_N_ITERATIONS == 0 and i != 0:
            if not os.path.exists(MODEL_DIRECTORY):
                os.makedirs(MODEL_DIRECTORY)
            tf.train.Saver().save(sess, MODEL_DIRECTORY + '/model-' + str(i) + '.cptk')
            print "Saved Model"
