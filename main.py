import tensorflow as tf

from constants import MODEL_DIRECTORY
from output_words import output_words
from setup_generator import setup_generator
from cross_platform_print import cross_platform_print

generator_stuff = setup_generator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cross_platform_print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)

    if not ckpt:
        cross_platform_print('Model not found! Have you run train.py for enough iterations for it to have saved a model yet?')
    else:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        output_words(sess, generator_stuff, "new")
