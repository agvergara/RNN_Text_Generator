#coding:cp1252

"""
@author: Antonio Gomez Vergara
"""

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, LSTM, Bidirectional, Embedding
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import random
import numpy as np
import sys
import tensorflow.python.util.deprecation as deprecation


class TextGeneratorModel():
   def __init__(self, path_checkpoints, x_test, x_train, seq_len, word_to_index, index_to_word, diversity, n_epochs, total_words, seed):
      deprecation._PRINT_DEPRECATION_WARNINGS = False
      self.path_checkpoint_file = path_checkpoints
      self.model = Sequential()
      self.callback_list = []
      self.X_test = x_test
      self.X_train = x_train
      self.seq_len = seq_len
      self.word_to_index = word_to_index
      self.index_to_word = index_to_word
      self.diversity_list = diversity
      self.n_epochs = n_epochs
      self.total_words = total_words
      self.seed = seed

   def build_model(self, input_dim, lstm_units, keep_prob=0.8, output_dim=1024):
      rate = 1 - keep_prob
      self.model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
      self.model.add(Bidirectional(LSTM(lstm_units)))
      if (keep_prob > 0):
         self.model.add(Dropout(rate=rate))
      self.model.add(Dense(input_dim))
      self.model.add(Activation('softmax'))

   def build_model_one_hot(self, input_dim, lstm_units=128, keep_prob=0.8):
      rate = 1 - keep_prob
      self.model.add(Bidirectional(LSTM(lstm_units), input_shape=(self.seq_len, input_dim)))
      if (rate < 1):
         self.model.add(Dropout(rate=rate))
      self.model.add(Dense(input_dim))
      self.model.add(Activation('softmax'))

   @staticmethod
   def config_adam_optimizer(learning_rate=0.005, beta_1=0.9, beta_2=0.999, decay=0):
      optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
      return optimizer

   @staticmethod
   def config_rmsprop_optimizer(learning_rate=0.01):
      optimizer = RMSprop(lr=learning_rate)
      return optimizer

   def config_callbacks(self, use_checkpoint=False, use_lambda_callback=False, use_early_stop=False, one_hot_flag=False):
      if(use_checkpoint):
         self.callback_list.append(ModelCheckpoint(self.path_checkpoint_file, monitor='acc', save_best_only=True, period=self.n_epochs))
      if(use_lambda_callback):
         if(one_hot_flag):
            self.callback_list.append(LambdaCallback(on_epoch_end=self.on_epoch_end_char))
         else:
            self.callback_list.append(LambdaCallback(on_epoch_end=self.on_epoch_end_word))
      if(use_early_stop):
         self.callback_list.append(EarlyStopping(monitor='loss', patience=10))

# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
   @staticmethod
   def sample(preds, temperature=1.0):
      # helper function to sample an index from a probability array
      preds = np.asarray(preds).astype('float64')
      preds = np.log(preds) / temperature
      exp_preds = np.exp(preds)
      preds = exp_preds / np.sum(exp_preds)
      probas = np.random.multinomial(1, preds, 1)
      return np.argmax(probas)

   def on_epoch_end_word(self, epoch, _):
      word_gen = 2000
      # Function invoked at end of each epoch. Prints generated text.
      #start_index = random.randint(0, len(self.X_train+self.X_test))
      #seed = (self.X_train+self.X_test)[start_index]
      curr_epoch = epoch + 1
      rate = curr_epoch % (self.n_epochs / 10)
      output_file = ".\\outputs\\Word_Embeddings_Epoch_{}_Diversity_{}.txt".format(curr_epoch, self.diversity_list[0])
      if (rate == 0):
         print('----- Epoch {} reached, generating text'.format(epoch + 1))
         with open (output_file, 'w') as fd:
            for diversity in self.diversity_list: #[0.8, 1.0, 1.2, 1.4]:
               sentence = self.seed
               sentence = sentence.split(" ")
               fd.write('----- Diversity: {}'.format(diversity))
               generated = ''
               #generated += ' '.join(sentence)
               generated += ' '.join(sentence)
               fd.write('----- Generating with seed: "' + generated + '"\n')
               fd.write('-----------------------------------------\n')
               #sys.stdout.write(generated)
               fd.write(generated)
               for i in range(word_gen):
                  x_pred = np.zeros((1, self.seq_len))
                  for idx, word in enumerate(sentence):
                     x_pred[0, idx] = self.word_to_index[word]

                  preds = self.model.predict(x_pred, verbose=0)[0]
                  next_index = self.sample(preds, diversity)
                  next_word = self.index_to_word[next_index]

                  sentence = sentence[1:]
                  sentence.append(next_word)

                  #sys.stdout.write(' '+next_word)
                  #sys.stdout.flush()
                  fd.write(' '+next_word)
                  fd.flush()

   def on_epoch_end_char(self, epoch, _):
      char_gen = 20000
      curr_epoch = epoch + 1
      rate = curr_epoch % (self.n_epochs / 10)
      output_file = ".\\outputs\\One_Hot_Epoch_{}_Diversity_{}.txt".format(curr_epoch, self.diversity_list[0])
      if (rate == 0):
         print('----- Epoch {} reached, generating text'.format(epoch + 1))
         with open (output_file, 'w') as fd:
            for diversity in self.diversity_list:
               sentence = self.seed
               fd.write('----- Diversity: {}'.format(diversity))
               generated = ''
               generated += sentence
               fd.write('----- Generating with seed: "' + generated + '"\n')
               fd.write('-----------------------------------------\n')
               fd.write(generated)
               for i in range(char_gen):
                  x_pred = np.zeros((1, self.seq_len, len(self.total_words)))
                  for idx, char in enumerate(sentence):
                     x_pred[0, idx, self.word_to_index[char]] = 1

                  preds = self.model.predict(x_pred, verbose=0)[0]
                  next_index = self.sample(preds, diversity)
                  next_char = self.index_to_word[next_index]

                  sentence = sentence[1:] + next_char

                  fd.write(next_char)
                  fd.flush()


