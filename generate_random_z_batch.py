import numpy as np

from constants import Z_SIZE, BATCH_SIZE


def generate_random_z_batch():
    return np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, Z_SIZE]).astype(np.float32)
