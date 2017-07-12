import math
import numpy as np
import os

from constants import SAMPLE_DIRECTORY, BATCH_SIZE_SAMPLE, DATA_SIZE_SQRT
from generate_random_z_batch import generate_random_z_batch
from save_images import save_images


def output_words(sess, z_in, Gz, name, batch_size):
    newZ = sess.run(Gz, feed_dict={z_in: generate_random_z_batch(batch_size)})

    if not os.path.exists(SAMPLE_DIRECTORY):
        os.makedirs(SAMPLE_DIRECTORY)

    square_root_batch = int(math.sqrt(BATCH_SIZE_SAMPLE))
    save_images(
        np.reshape(
            newZ[0:BATCH_SIZE_SAMPLE],
            [BATCH_SIZE_SAMPLE, DATA_SIZE_SQRT, DATA_SIZE_SQRT]
        ),
        [square_root_batch, square_root_batch],
        SAMPLE_DIRECTORY + '/' + name + '.png'
    )
