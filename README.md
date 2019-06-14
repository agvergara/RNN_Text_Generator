# RNN_Text_Generator
Given a corpus, it generates text based on a seed using one-hot or word embeddings.
There's a lot of work to be done: improve the model, tune better the parameters, etc.
I'm still learning the workarounds of Recurent Neural Networks, these type of networks are hard to understand for me.

# MODELS
They are practically the same model, but with an embedding layer or without it.
### One_hot_model

```
__________________________________________________________________________
Layer (type)                           Output Shape              Param #
==========================================================================
bidirectional_1 (Bidirectional LSTM)   (None, 256)               166912
__________________________________________________________________________
dropout_1 (Dropout)                    (None, 256)               0
__________________________________________________________________________
dense_1 (Dense)                        (None, 34)                8738
__________________________________________________________________________
activation_1 (Activation)              (None, 34)                0
==========================================================================
Total params: 175,650
Trainable params: 175,650
Non-trainable params: 0
__________________________________________________________________________
```

### Word_embedding_model

```
__________________________________________________________________________
Layer (type)                             Output Shape              Param #
==========================================================================
embedding_1 (Embedding)                (None, None, 1024)        13263872
__________________________________________________________________________
bidirectional_1 (Bidirectional LSTM)   (None, 256)               166912
__________________________________________________________________________
dropout_1 (Dropout)                    (None, 256)               0
__________________________________________________________________________
dense_1 (Dense)                        (None, 12953)             3328921
__________________________________________________________________________
activation_1 (Activation)              (None, 12953)             0
==========================================================================
Total params: 17,773,465
Trainable params: 17,773,465
Non-trainable params: 0
__________________________________________________________________________
```

## Usage:
The usage if kind of simple, just import the class and clean the data!

```python

import sys
import text_utils
from rnn_class import TextGeneratorModel

MIN_WORD_FREQ = 10
SEED = "Your seed here"
SEQUENCE_LEN = len(SEED)
STRIDE = 1
BATCH_SIZE = 32
TEST_PERCENTAGE = 2
CHECKPOINT_FILE = "./checkpoints/LSTM_GEN_word_embeddings_epoch_{epoch:03d}"
EPOCHS = 100

corpus_path = ".\\corpus.txt"
corpus_length_chars, full_text, corpus_length_words, words_in_corpus = text_utils.get_corpus_words(corpus_path)

num_ignored_words, ignored_words, word_to_index, index_to_word, words_not_ignored, total_words = text_utils.calc_word_frequency(words_in_corpus, full_text, SEED.lower(), MIN_WORD_FREQ)

sequences, next_words, sequences_ignored = text_utils.check_redundancy(words_in_corpus, ignored_words, SEQUENCE_LEN, STRIDE)

x_train, x_test, y_train, y_test = text_utils.shuffle_split_train_test(sequences, next_words, TEST_PERCENTAGE)

#Model configuration

diversity = [0.4, 0.8, 1.0, 1.2, 1.4]

model = TextGeneratorModel(CHECKPOINT_FILE, x_test, x_train, SEQUENCE_LEN, word_to_index, index_to_word, diversity, EPOCHS, total_words, SEED.lower())

input_dim = len(words_not_ignored)
model.build_model(input_dim, lstm_units=128, keep_prob=0.8, output_dim=1024)

optimizer = model.config_rmsprop_optimizer(learning_rate=0.001)

model.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,  metrics=["accuracy"])

model.config_callbacks(use_checkpoint=True, use_lambda_callback=True, use_early_stop=False)

steps_per_epoch = int(len(x_train) / BATCH_SIZE) + 1
validate_steps = int(len(x_test) / BATCH_SIZE) + 1
model.model.fit_generator(generator=text_utils.vectorization(x_train, y_train, BATCH_SIZE, word_to_index, SEQUENCE_LEN),
                          steps_per_epoch=steps_per_epoch,
                          epochs=EPOCHS,
                          callbacks=model.callback_list,
                          validation_data=text_utils.vectorization(x_test, y_test, BATCH_SIZE, word_to_index, SEQUENCE_LEN),
                          validation_steps=validate_steps)
```

