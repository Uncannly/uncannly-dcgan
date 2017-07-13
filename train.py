import os
import tensorflow as tf

import input_data
from constants import MODEL_DIRECTORY
from generate_random_z_batch import generate_random_z_batch
from output_words import output_words
from setup_discriminators import setup_discriminators
from setup_generator import setup_generator
from cross_platform_print import cross_platform_print

data_sets = input_data.read_data_sets()
tf.reset_default_graph()

generator_stuff = setup_generator()
Gz = generator_stuff['Gz']
theta_G = generator_stuff['theta_G']
z_in = generator_stuff['z_in']

discriminator_stuff = setup_discriminators(Gz)
theta_D = discriminator_stuff['theta_D']
d_loss = discriminator_stuff['d_loss']
g_loss = discriminator_stuff['g_loss']
X = discriminator_stuff['X']

D_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_G)

NUM_ITERATIONS = 500000
OUTPUT_WORDS_EVERY_N_ITERATIONS = 10
SAVE_MODEL_EVERY_N_ITERATIONS = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_ITERATIONS):
        X_mb = data_sets.next_batch()

        _, d_loss_curr = sess.run([D_solver, d_loss], feed_dict={X: X_mb, z_in: generate_random_z_batch()})
        _, g_loss_curr = sess.run([G_solver, g_loss], feed_dict={z_in: generate_random_z_batch()})

        if i % OUTPUT_WORDS_EVERY_N_ITERATIONS == 0:
            cross_platform_print("Gen Loss: " + str(g_loss_curr) + " Disc Loss: " + str(d_loss_curr))
            output_words(sess, generator_stuff, str(i))

        if i % SAVE_MODEL_EVERY_N_ITERATIONS == 0 and i != 0:
            if not os.path.exists(MODEL_DIRECTORY):
                os.makedirs(MODEL_DIRECTORY)
            tf.train.Saver().save(sess, MODEL_DIRECTORY + '/model-' + str(i) + '.cptk')
            cross_platform_print("Saved Model")
