import tensorflow as tf

from constants import MODEL_DIRECTORY, BATCH_SIZE_SAMPLE
from output_words import output_words
from setup_generator import setup_generator

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
        output_words(sess, z_in, Gz, "new", BATCH_SIZE_SAMPLE)
