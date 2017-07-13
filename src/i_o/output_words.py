import os

from src.constants import SAMPLE_DIRECTORY, MAX_WORD_LENGTH, COUNT_AVAILABLE_LETTERS, LETTERS
from src.gan.generate_random_z_batch import generate_random_z_batch

OUTPUT_SAMPLE_SIZE = 10

def output_words(sess, generator_stuff, name):
    newZ = sess.run(generator_stuff['Gz'], feed_dict={generator_stuff['z_in']: generate_random_z_batch()})

    if not os.path.exists(SAMPLE_DIRECTORY):
        os.makedirs(SAMPLE_DIRECTORY)

    with open(SAMPLE_DIRECTORY + '/' + name + '.txt', 'w') as f:
        for i in range(OUTPUT_SAMPLE_SIZE):
            this_word = newZ[i]

            by_letter = this_word.reshape([MAX_WORD_LENGTH, COUNT_AVAILABLE_LETTERS])
            this_word_as_string = ''
            for word_position in by_letter:

                index_of_most_likely_letter_at_this_word_position = None
                max = 0
                for i in range(len(word_position)):
                    if word_position[i] > max:
                        max = word_position[i]
                        index_of_most_likely_letter_at_this_word_position = i

                this_letter = LETTERS[index_of_most_likely_letter_at_this_word_position]

                if this_letter != '$':
                    this_word_as_string = this_word_as_string + this_letter


            f.write(this_word_as_string)
