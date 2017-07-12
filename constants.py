SAMPLE_DIRECTORY = './figs'
MODEL_DIRECTORY = './models'

Z_SIZE = 100

BATCH_SIZE = 100

DATA_SIZE_SQRT = 32
DATA_SIZE = DATA_SIZE_SQRT * DATA_SIZE_SQRT

# convolution requires data of at least 2 dimensions.
# think of the the sliding window.
# so it is necessary to have this be something square.
# still pondering how I will get spelling + pronunciation data into this form...
# or if convolution is even the proper strategy for this domain

# re: 32, something about the discriminator's magic numbers
# very specifically requires this to be 32.