#coding:cp1252

"""
@author: Antonio Gomez Vergara
Generates text using word embeddings (still in process of study how word embeddings works)
"""

import sys
import text_utils
from rnn_class import TextGeneratorModel

MIN_WORD_FREQ = 10
SEED = "Nenita hello"
SEQUENCE_LEN = len(SEED)
STRIDE = 1
BATCH_SIZE = 2048
TEST_PERCENTAGE = 2
CHECKPOINT_FILE = "./checkpoints/LSTM_GEN_word_embeddings_epoch_{epoch:03d}"
EPOCHS = 100

corpus_path = ".\\corpus.txt"
corpus_length_chars, full_text, corpus_length_words, words_in_corpus = text_utils.get_corpus_words(corpus_path)

print("Corpus number of chars -> {}".format(corpus_length_chars))
print("Corpus number of words -> {}".format(corpus_length_words))

num_ignored_words, ignored_words, word_to_index, index_to_word, words_not_ignored, total_words = text_utils.calc_word_frequency(words_in_corpus, full_text, SEED.lower(), MIN_WORD_FREQ)

print("Calculating word frequency. . .")
print("Ignoring words with less than {} frequency".format(MIN_WORD_FREQ))
print("Number of words ignored -> {}".format(num_ignored_words))
print("Number of words after ignoring -> {}".format(len(words_not_ignored)))

sequences, next_words, sequences_ignored = text_utils.check_redundancy(words_in_corpus, ignored_words, SEQUENCE_LEN, STRIDE)
print("Deleting redundant sequences. . .")
print("Sequences ignored -> {} sequences".format(sequences_ignored))

x_train, x_test, y_train, y_test = text_utils.shuffle_split_train_test(sequences, next_words, TEST_PERCENTAGE)
print("Shuffling the sequences and split it into Test({}%)/Train({}%)".format((100-TEST_PERCENTAGE), TEST_PERCENTAGE))
print("Size of Test set -> {}".format(len(x_test)))
print("Size of Train set -> {}".format(len(x_train)))

#Model configuration
print("Configuring model. . .")

diversity = [1.4]

model = TextGeneratorModel(CHECKPOINT_FILE, x_test, x_train, SEQUENCE_LEN, word_to_index, index_to_word, diversity, EPOCHS, total_words, SEED.lower())

input_dim = len(words_not_ignored)
model.build_model(input_dim, lstm_units=128, keep_prob=0.8, output_dim=1024)

optimizer = model.config_rmsprop_optimizer(learning_rate=0.001)

model.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,  metrics=["accuracy"])
print("Compiling model. . .")
print('\n')

model.config_callbacks(use_checkpoint=True, use_lambda_callback=True, use_early_stop=False)

steps_per_epoch = int(len(x_train) / BATCH_SIZE) + 1
validate_steps = int(len(x_test) / BATCH_SIZE) + 1
model.model.fit_generator(generator=text_utils.vectorization(x_train, y_train, BATCH_SIZE, word_to_index, SEQUENCE_LEN),
                          steps_per_epoch=steps_per_epoch,
                          epochs=EPOCHS,
                          callbacks=model.callback_list,
                          validation_data=text_utils.vectorization(x_test, y_test, BATCH_SIZE, word_to_index, SEQUENCE_LEN),
                          validation_steps=validate_steps)

