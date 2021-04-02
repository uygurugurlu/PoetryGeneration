import pandas as pd
import tensorflow as tf
import numpy as np

dataset = pd.read_csv("sequence_data.csv").to_numpy()
dataset = np.delete(dataset, 0, axis=1)
xtrain = dataset[:, :-1]
labels = dataset[:, -1]
print(labels)
print(xtrain)
ys = tf.keras.utils.to_categorical(labels, num_classes=15309)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(15309, 240, input_length=14))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(15309, activation='softmax'))
adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
for i in range(20, 101, 20):
    history = model.fit(xtrain, ys, epochs=i, verbose=1)
    model.save("poetryModel" + str(i))
    print("Model with epoch ", str(i), " is saved")
