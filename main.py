import tensorflow as tf
import numpy as np
import os

from generator import generator
from save_images import save_images
from constants import SAMPLE_DIRECTORY, MODEL_DIRECTORY, Z_SIZE

initializer = tf.truncated_normal_initializer(stddev=0.02)
z_in = tf.placeholder(shape=[None,Z_SIZE],dtype=tf.float32)
Gz = generator(z_in, initializer)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

batch_size_sample = 36

with tf.Session() as sess:
    sess.run(init)
    print 'Loading Model...'
    ckpt = tf.train.get_checkpoint_state(MODEL_DIRECTORY)

    if not ckpt:
        print 'Model not found! Have you not run train.py for 10 iterations yet?'
    else:
        saver.restore(sess,ckpt.model_checkpoint_path)

        z2 = np.random.uniform(-1.0,1.0,size=[batch_size_sample,Z_SIZE]).astype(np.float32) #Generate a random z batch
        newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
        if not os.path.exists(SAMPLE_DIRECTORY):
            os.makedirs(SAMPLE_DIRECTORY)
        save_images(np.reshape(newZ[0:batch_size_sample],[36,32,32]),[6,6],SAMPLE_DIRECTORY+'/figMain.png')
