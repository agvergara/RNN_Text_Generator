# RNN_Text_Generator
Given a corpus, it generates text based on a seed using one-hot or word embeddings.
There's a lot of work to be done: improve the model, tune better the parameters, etc.
I'm still learning the workarounds of Recurent Neural Networks, these type of networks are hard to understand for me.

# MODELS
They are practically the same model, but with an embedding layer or without it.
### One_hot_model

```javascript
{
    "class_name": "Sequential",
    "config": {
        "name": "sequential_1",
        "layers": [{
            "class_name": "Bidirectional",
            "config": {
                "name": "bidirectional_1",
                "trainable": true,
                "batch_input_shape": [null, 54, 34],
                "dtype": "float32",
                "layer": {
                    "class_name": "LSTM",
                    "config": {
                        "name": "lstm_1",
                        "trainable": true,
                        "return_sequences": false,
                        "return_state": false,
                        "go_backwards": false,
                        "stateful": false,
                        "unroll": false,
                        "units": 128,
                        "activation": "tanh",
                        "recurrent_activation": "hard_sigmoid",
                        "use_bias": true,
                        "kernel_initializer": {
                            "class_name": "VarianceScaling",
                            "config": {
                                "scale": 1.0,
                                "mode": "fan_avg",
                                "distribution": "uniform",
                                "seed": null
                            }
                        },
                        "recurrent_initializer": {
                            "class_name": "Orthogonal",
                            "config": {
                                "gain": 1.0,
                                "seed": null
                            }
                        },
                        "bias_initializer": {
                            "class_name": "Zeros",
                            "config": {}
                        },
                        "unit_forget_bias": true,
                        "kernel_regularizer": null,
                        "recurrent_regularizer": null,
                        "bias_regularizer": null,
                        "activity_regularizer": null,
                        "kernel_constraint": null,
                        "recurrent_constraint": null,
                        "bias_constraint": null,
                        "dropout": 0.0,
                        "recurrent_dropout": 0.0,
                        "implementation": 1
                    }
                },
                "merge_mode": "concat"
            }
        }, {
            "class_name": "Dropout",
            "config": {
                "name": "dropout_1",
                "trainable": true,
                "rate": 0.19999999999999996,
                "noise_shape": null,
                "seed": null
            }
        }, {
            "class_name": "Dense",
            "config": {
                "name": "dense_1",
                "trainable": true,
                "units": 34,
                "activation": "linear",
                "use_bias": true,
                "kernel_initializer": {
                    "class_name": "VarianceScaling",
                    "config": {
                        "scale": 1.0,
                        "mode": "fan_avg",
                        "distribution": "uniform",
                        "seed": null
                    }
                },
                "bias_initializer": {
                    "class_name": "Zeros",
                    "config": {}
                },
                "kernel_regularizer": null,
                "bias_regularizer": null,
                "activity_regularizer": null,
                "kernel_constraint": null,
                "bias_constraint": null
            }
        }, {
            "class_name": "Activation",
            "config": {
                "name": "activation_1",
                "trainable": true,
                "activation": "softmax"
            }
        }]
    },
    "keras_version": "2.2.4",
    "backend": "tensorflow"
}
```

### Word_embedding_model

```javascript
{
    "class_name": "Sequential",
    "config": {
        "name": "sequential_1",
        "layers": [{
            "class_name": "Embedding",
            "config": {
                "name": "embedding_1",
                "trainable": true,
                "batch_input_shape": [null, null],
                "dtype": "float32",
                "input_dim": 12953,
                "output_dim": 1024,
                "embeddings_initializer": {
                    "class_name": "RandomUniform",
                    "config": {
                        "minval": -0.05,
                        "maxval": 0.05,
                        "seed": null
                    }
                },
                "embeddings_regularizer": null,
                "activity_regularizer": null,
                "embeddings_constraint": null,
                "mask_zero": false,
                "input_length": null
            }
        }, {
            "class_name": "Bidirectional",
            "config": {
                "name": "bidirectional_1",
                "trainable": true,
                "layer": {
                    "class_name": "LSTM",
                    "config": {
                        "name": "lstm_1",
                        "trainable": true,
                        "return_sequences": false,
                        "return_state": false,
                        "go_backwards": false,
                        "stateful": false,
                        "unroll": false,
                        "units": 128,
                        "activation": "tanh",
                        "recurrent_activation": "hard_sigmoid",
                        "use_bias": true,
                        "kernel_initializer": {
                            "class_name": "VarianceScaling",
                            "config": {
                                "scale": 1.0,
                                "mode": "fan_avg",
                                "distribution": "uniform",
                                "seed": null
                            }
                        },
                        "recurrent_initializer": {
                            "class_name": "Orthogonal",
                            "config": {
                                "gain": 1.0,
                                "seed": null
                            }
                        },
                        "bias_initializer": {
                            "class_name": "Zeros",
                            "config": {}
                        },
                        "unit_forget_bias": true,
                        "kernel_regularizer": null,
                        "recurrent_regularizer": null,
                        "bias_regularizer": null,
                        "activity_regularizer": null,
                        "kernel_constraint": null,
                        "recurrent_constraint": null,
                        "bias_constraint": null,
                        "dropout": 0.0,
                        "recurrent_dropout": 0.0,
                        "implementation": 1
                    }
                },
                "merge_mode": "concat"
            }
        }, {
            "class_name": "Dropout",
            "config": {
                "name": "dropout_1",
                "trainable": true,
                "rate": 0.19999999999999996,
                "noise_shape": null,
                "seed": null
            }
        }, {
            "class_name": "Dense",
            "config": {
                "name": "dense_1",
                "trainable": true,
                "units": 12953,
                "activation": "linear",
                "use_bias": true,
                "kernel_initializer": {
                    "class_name": "VarianceScaling",
                    "config": {
                        "scale": 1.0,
                        "mode": "fan_avg",
                        "distribution": "uniform",
                        "seed": null
                    }
                },
                "bias_initializer": {
                    "class_name": "Zeros",
                    "config": {}
                },
                "kernel_regularizer": null,
                "bias_regularizer": null,
                "activity_regularizer": null,
                "kernel_constraint": null,
                "bias_constraint": null
            }
        }, {
            "class_name": "Activation",
            "config": {
                "name": "activation_1",
                "trainable": true,
                "activation": "softmax"
            }
        }]
    },
    "keras_version": "2.2.4",
    "backend": "tensorflow"
}
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

