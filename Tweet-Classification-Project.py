import numpy as np
import os
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional,GRU,SpatialDropout1D
from tensorflow.keras.utils import plot_model

# Data processing as before
dataframe = pd.read_csv("train.csv")
dataframe = dataframe[["text", "target"]]
print(dataframe.shape)

plt.figure(figsize=(10, 6))
sns.countplot(x="target", data=dataframe)
plt.title('The Distribution of Disaster and Not Disaster Tweets')
plt.show()

x = dataframe['text']
y = dataframe['target']
x_rest, x_test, y_rest, y_test = train_test_split(x, y, test_size=0.1, random_state=434)
x_train, x_validation, y_train, y_validation = train_test_split(x_rest, y_rest, test_size=0.1, random_state=434)
vocab_size = 10000
trunc_type = 'post'
padding_type = 'post'
tokenizer = Tokenizer(num_words=vocab_size, 
                      char_level=False)

tokenizer.fit_on_texts(x_train)
training_sequences = tokenizer.texts_to_sequences(x_train)
sequence_lengths = [len(seq) for seq in training_sequences]
max_len = max(sequence_lengths)

word_index = tokenizer.word_index
total_words = len(word_index)
print(total_words)
vocab_size = total_words
training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences,
                               maxlen=max_len,
                               padding=padding_type,
                               truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(x_validation)
validation_padded = pad_sequences(validation_sequences,
                                  maxlen=max_len,
                                  padding=padding_type,
                                  truncating=trunc_type)


embedding_index = {}
glove_path = "glove.6B.300d.txt"
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefficients


embedding_dim = 300  
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32') 
    y_pred = tf.round(y_pred) 
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    precision = tp / (tf.reduce_sum(tf.cast(y_pred, 'float'), axis=0) + tf.keras.backend.epsilon())
    recall = tp / (tf.reduce_sum(tf.cast(y_true, 'float'), axis=0) + tf.keras.backend.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

model_gru = Sequential()
model_gru.add(Embedding(vocab_size,
                     embedding_dim,
                     weights=[embedding_matrix],
                     input_length = max_len))
model_gru.add(GRU(64, return_sequences = False))
model_gru.add(Dropout(0.3))
model_gru.add(Dense(1, activation = 'sigmoid'))
model_gru.compile(loss = 'binary_crossentropy',
                       optimizer = 'adam',
                       metrics=[f1_score])
model_gru.build(input_shape=(None, max_len))
num_epochs = 10
early_stop = EarlyStopping(monitor='val_loss', patience=1)
history = model_gru.fit(training_padded,
                     y_train,
                     epochs=num_epochs,
                     batch_size = 128, 
                     validation_data=(validation_padded, y_validation),
                     callbacks =[early_stop],
                     verbose=2)

model_gru.evaluate(testing_padded, y_test, batch_size = 128)




plt.figure(figsize=(10, 6))
plt.plot(history.history['f1_score'], label='Training F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title('Training vs Validation F1 Score- GRU')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()