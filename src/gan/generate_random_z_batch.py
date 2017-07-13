import numpy as np

from src.constants import Z_SIZE, BATCH_SIZE


def generate_random_z_batch():
    return np.random.uniform(-1., 1., size=[BATCH_SIZE, Z_SIZE])
