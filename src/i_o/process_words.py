from src.constants import LETTERS, COUNT_AVAILABLE_LETTERS, DATA_SIZE


def process_words():
    word_spellings = []

    with open('src/i_o/words.txt', 'r') as f:
        for line in f:
            spelling_matrix = [0] * DATA_SIZE
            for word_position, letter in enumerate(line):

                letter_index = None
                for index_of_letter_in_letters, letter_in_letters in enumerate(LETTERS):
                    if letter_in_letters == letter:
                        letter_index = index_of_letter_in_letters

                    word_position_offset = word_position * COUNT_AVAILABLE_LETTERS

                spelling_matrix_index = None
                if letter_index is not None:
                    spelling_matrix_index = word_position_offset + letter_index
                else:
                    spelling_matrix_index = word_position_offset + COUNT_AVAILABLE_LETTERS - 1

                if spelling_matrix_index < len(spelling_matrix):
                    spelling_matrix[spelling_matrix_index] = 1

        word_spellings.append(spelling_matrix)

    return word_spellings
