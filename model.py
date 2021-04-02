import pandas as pd
import tensorflow as tf
import numpy as np
dataset = pd.read_csv("sequence_data.csv").to_numpy()
dataset= np.delete(dataset, 0, axis=1)
xtrain = dataset[:, :-1]
labels = dataset[:, -1]
print(labels)
print(xtrain)
ys = tf.keras.utils.to_categorical(labels, num_classes=15293)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(15293, 240, input_length=11))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(15293, activation='softmax'))
adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xtrain, ys, epochs=40, verbose=1)
model.save("poetryModel2")
