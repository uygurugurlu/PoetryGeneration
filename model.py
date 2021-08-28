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
model.add(tf.keras.layers.Embedding(15309, 60, input_length=14))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(15309, activation='softmax'))
adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# for i in range(20, 101, 20):
#     history = model.fit(xtrain, ys, epochs=i, verbose=1)
#     model.save("poetryModel" + str(i))
#     print("Model with epoch ", str(i), " is saved")

history = model.fit(xtrain, ys, epochs=2, verbose=1)
model.save("poetryModelWithlr005")
#lr= 0.003, acc = 0.2945 epochs=5, embedding 240
#lr= 0.005, acc = 0.2742
#lr= 0.01, acc = 0.1845
#lr= 0.015, acc = 0.1718

#embedding 60 epoch=2 lr=0.01 acc = 0.1982
#embedding 120 epoch=2 lr=0.01 acc = 0.1980
#embedding 240 epoch=2 lr=0.01 acc = 0.1878
#embedding 240 epoch=2 lr=0.01 acc = 0.1719