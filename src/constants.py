SAMPLE_DIRECTORY = './figs'
MODEL_DIRECTORY = './models'

Z_SIZE = 50

BATCH_SIZE = 100
THIS_MAGIC_NUMBER = 128
MAX_WORD_LENGTH = len('SUPERCALIFRAGILISTICEXPIALIDOCIOUS') + 1

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', '\'', '-', '\n', '$']
COUNT_AVAILABLE_LETTERS = len(LETTERS)

DATA_SIZE = MAX_WORD_LENGTH * COUNT_AVAILABLE_LETTERS
