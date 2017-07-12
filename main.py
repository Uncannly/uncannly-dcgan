import numpy as np
import os
import tensorflow as tf

from constants import SAMPLE_DIRECTORY, MODEL_DIRECTORY, BATCH_SIZE_SAMPLE
from save_images import save_images
from setup_generator import setup_generator
from generate_random_z_batch import generate_random_z_batch

generator_stuff = setup_generator()
z_in = generator_stuff['z_in']
initializer = generator_stuff['initializer']
Gz = generator_stuff['Gz']

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print 'Loading Model...'
    ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)

    if not ckpt:
        print 'Model not found! Have you not run train.py for 10 iterations yet?'
    else:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        z2 = generate_random_z_batch(BATCH_SIZE_SAMPLE)
        newZ = sess.run(Gz, feed_dict={z_in: z2})  # Use new z to get sample images from generator.
        if not os.path.exists(SAMPLE_DIRECTORY):
            os.makedirs(SAMPLE_DIRECTORY)
        save_images(np.reshape(newZ[0:BATCH_SIZE_SAMPLE], [BATCH_SIZE_SAMPLE, 32, 32]), [6, 6], SAMPLE_DIRECTORY + '/figMain.png')
