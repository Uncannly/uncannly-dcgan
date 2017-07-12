import os

from constants import SAMPLE_DIRECTORY
from generate_random_z_batch import generate_random_z_batch

OUTPUT_SAMPLE_SIZE = 10

def output_words(sess, z_in, Gz, name):
    newZ = sess.run(Gz, feed_dict={z_in: generate_random_z_batch()})

    if not os.path.exists(SAMPLE_DIRECTORY):
        os.makedirs(SAMPLE_DIRECTORY)

    with open(SAMPLE_DIRECTORY + '/' + name + '.txt', 'w') as f:
        for i in range(OUTPUT_SAMPLE_SIZE):
            f.write(str(newZ[i]))
