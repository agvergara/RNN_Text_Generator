#coding:cp1252

"""
@author: Antonio Gomez Vergara
Generates text using one_hot word embeddings
"""
import sys
import text_utils
from rnn_class import TextGeneratorModel

SEED = "In the beginning God created the heavens and the earth"
SEQUENCE_LEN = len(SEED)
STRIDE = 3
BATCH_SIZE = 128
CHECKPOINT_FILE = "./checkpoints/LSTM_GEN_one_hot_epoch_{epoch:03d}.model"
EPOCHS = 100
USE_ONE_HOT = True

corpus_path = ".\\corpus.txt"
corpus_length_chars, full_text, corpus_length_words, words_in_corpus = text_utils.get_corpus_words(corpus_path)

print("Corpus number of chars -> {}".format(corpus_length_chars))
print("Corpus number of words -> {}".format(corpus_length_words))

_, _, word_to_index, index_to_word, _, total_words = text_utils.calc_word_frequency(words_in_corpus, full_text, SEED, one_hot_flag=USE_ONE_HOT)
print("Calculating word frequency. . .")

sequences, next_words, _ = text_utils.check_redundancy(full_text, _, SEQUENCE_LEN, STRIDE, one_hot_flag=USE_ONE_HOT)

#Model Configuration
diversity = [1.4]

model = TextGeneratorModel(CHECKPOINT_FILE, _, _, SEQUENCE_LEN, word_to_index, index_to_word, diversity, EPOCHS, total_words, SEED.lower())

input_dim = len(total_words)

model.build_model_one_hot(input_dim, lstm_units=128, keep_prob=0.8)
optimizer = model.config_rmsprop_optimizer(learning_rate=0.01)
model.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Compiling model. . .")
print('\n')

model.config_callbacks(use_checkpoint=True, use_lambda_callback=True, use_early_stop=False, one_hot_flag=USE_ONE_HOT)

x, y = text_utils.vectorization_one_hot(sequences, next_words, total_words, word_to_index, SEQUENCE_LEN)
model.model.fit(x, y,
               batch_size=128,
               epochs=EPOCHS,
               callbacks=model.callback_list)
