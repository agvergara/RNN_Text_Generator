#coding:cp1252

"""
@author: Antonio Gomez Vergara
"""

from io import open
import numpy as np

def get_corpus_words(corpus):
   corpus_length_chars = 0
   corpus_length_words = 0

   with open(corpus, 'r', encoding='latin-1') as fd:
      full_text = fd.read().lower().replace('\n', ' \n ')

   words_in_corpus = [word for word in full_text.split(' ') if word.strip() != '' or word == '\n']

   corpus_length_chars = len(full_text)
   corpus_length_words = len(words_in_corpus)

   return corpus_length_chars, full_text, corpus_length_words, words_in_corpus

def calc_word_frequency(words_in_corpus, full_text, seed, min_word_freq=0, one_hot_flag=False):
   num_ignored_words = 0
   word_frequency = {}
   word_to_index = {}
   index_to_word = {}
   ignored_words = set()

   for word in words_in_corpus:
      word_frequency[word] = word_frequency.get(word, 0) + 1

   for key, val in word_frequency.items():
      if word_frequency[key] < min_word_freq:
         ignored_words.add(key)

   total_words = set(words_in_corpus)
   words = sorted(set(total_words) - ignored_words)
   if (not one_hot_flag):
      seed = seed.split(" ")
      for diff_word in seed:
         if (diff_word not in words):
            words.append(diff_word)

      for index, context in enumerate(words):
         word_to_index[context] = index
         index_to_word[index] = context
   else:
      total_words = sorted(list(set(full_text)))
      for index, context in enumerate(total_words):
         word_to_index[context] = index
         index_to_word[index] = context
   num_ignored_words = len(words_in_corpus) - len(words)

   return num_ignored_words, ignored_words, word_to_index, index_to_word, words, total_words

def check_redundancy(text_words, sequences_ignored, sequence_len=10, stride=1, one_hot_flag=False):
   sequences = []
   next_words = []
   num_sequences_ignored = 0
   start = 0
   sequence_intersect = 0
   end = len(text_words) - sequence_len

   for idx in range(start, end, stride):
      if (not one_hot_flag):
         sequence_intersect = len(set(text_words[idx : idx+sequence_len+1]).intersection(sequences_ignored))
         if (not sequence_intersect):
            sequences.append(text_words[idx : idx+sequence_len])
            next_words.append(text_words[idx+sequence_len])
         else:
            num_sequences_ignored += 1
      else:
         sequences.append(text_words[idx : idx+sequence_len])
         next_words.append(text_words[idx+sequence_len])
   return sequences, next_words, num_sequences_ignored

def shuffle_split_train_test(sequences, next_words, test_percentage=5):
   aux_sequences = []
   aux_next_words = []

   for idx in np.random.permutation(len(sequences)):
      aux_sequences.append(sequences[idx])
      aux_next_words.append(next_words[idx])

   split_index = int(len(sequences) * (1-(test_percentage/100)))

   X_train, X_test = aux_sequences[:split_index], aux_sequences[split_index:]
   Y_train, Y_test = aux_next_words[:split_index], aux_next_words[split_index:]

   return X_train, X_test, Y_train, Y_test

def vectorization(X, Y, batch_size, word_to_index, sequence_len=10):
   index = 0
   while True:
      x = np.zeros((batch_size, sequence_len), dtype=np.int32)
      y = np.zeros((batch_size), dtype=np.int32)
      for idx in range(batch_size):
         sentence = X[index % len(X)]
         for pos, word in enumerate(sentence):
            x[idx, pos] = word_to_index[word]
         y[idx] = word_to_index[Y[index % len(X)]]
         index += 1
      yield x, y

def vectorization_one_hot(X, Y, total_words, word_to_index, sequence_len=10):
   x = np.zeros((len(X), sequence_len, len(total_words)), dtype=np.bool)
   y = np.zeros((len(X), len(total_words)), dtype=np.bool)
   for idx, sentence in enumerate(X):
      for pos, word in enumerate(sentence):
         x[idx, pos, word_to_index[word]] = 1
      y[idx, word_to_index[Y[idx]]] = 1
   return x, y