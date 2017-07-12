import numpy as np
from constants import Z_SIZE

def generate_random_z_batch(batch_size):
    return np.random.uniform(-1.0, 1.0, size=[batch_size, Z_SIZE]).astype(np.float32)
