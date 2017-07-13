import os

from src.constants import SAMPLE_DIRECTORY
from src.gan.generate_random_z_batch import generate_random_z_batch

OUTPUT_SAMPLE_SIZE = 10

def output_words(sess, generator_stuff, name):
    newZ = sess.run(generator_stuff['Gz'], feed_dict={generator_stuff['z_in']: generate_random_z_batch()})

    if not os.path.exists(SAMPLE_DIRECTORY):
        os.makedirs(SAMPLE_DIRECTORY)

    with open(SAMPLE_DIRECTORY + '/' + name + '.txt', 'w') as f:
        for i in range(OUTPUT_SAMPLE_SIZE):
            f.write(str(newZ[i]))
