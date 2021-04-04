import tensorflow as tf
import pandas as pd
import numpy as np
import random

from zemberek import (
    TurkishTokenizer,
)
from zemberek.tokenization.token import Token

model1 = tf.keras.models.load_model('poetryModel20')
model2 = tf.keras.models.load_model('poetryModel40')
model3 = tf.keras.models.load_model('poetryModel60')
model4 = tf.keras.models.load_model('poetryModel80')

seed_text = "selam ben şiir yazıyorum"
seed_text_original = seed_text
next_words = 100
max_length = 14
tokenizeArray = pd.read_csv("tokenizeList.csv")
tokenizeArray = tokenizeArray['0'].to_numpy().tolist()

dataset = pd.read_csv("sequence_data.csv").to_numpy()
dataset = np.delete(dataset, 0, axis=1).tolist()


def addToTokenizeArray(word):
    if word in tokenizeArray:
        return tokenizeArray.index(word)
    else:
        tokenizeArray.append(word)
        return tokenizeArray.index(word)


def applyPadding(tokenarray):
    newArray = []
    zeroAmount = max_length - len(tokenarray)
    zeroArray = []
    for i in range(zeroAmount):
        zeroArray.append(0)
    concat = zeroArray + tokenarray
    return concat


def addLineToDataset(tokenarray):
    indexed = []
    for i in range(len(tokenarray)):
        indexed.append(addToTokenizeArray(tokenarray[i].content))

    dataset.append(applyPadding(indexed))
    return applyPadding(indexed)


def addLineToDatasetForPrediction(tokenarray):
    indexed = []
    for i in range(len(tokenarray)):
        indexed.append(addToTokenizeArray(tokenarray[i].content))
    indexed.append(1)

    dataset.append(applyPadding(indexed))
    return applyPadding(indexed)


def deleteFirstWord(line):
    return (line.partition(' ')[2])


tokenizer = TurkishTokenizer.builder().accept_all().ignore_types([Token.Type.NewLine,
                                                                  Token.Type.SpaceTab]).build()
print("----------------------------------")
print("Model1")
poetry = seed_text
for ss in range(next_words):
    tokens = tokenizer.tokenize(seed_text)
    res = addLineToDataset(tokens)
    predicted = model1.predict_classes([res], verbose=0)

    word_list = seed_text.split()
    number_of_words = len(word_list)
    if number_of_words >= max_length:
        seed_text = deleteFirstWord(seed_text)

    poetry += " " + tokenizeArray[predicted[0]]
    seed_text = seed_text + " " + tokenizeArray[predicted[0]]

print(poetry)
poetry = seed_text_original
seed_text = seed_text_original

print("----------------------------------")
print("Model2")
poetry = seed_text
for ss in range(next_words):
    tokens = tokenizer.tokenize(seed_text)
    res = addLineToDataset(tokens)
    predicted = model2.predict_classes([res], verbose=0)

    word_list = seed_text.split()
    number_of_words = len(word_list)
    if number_of_words >= max_length:
        seed_text = deleteFirstWord(seed_text)

    poetry += " " + tokenizeArray[predicted[0]]
    seed_text = seed_text + " " + tokenizeArray[predicted[0]]

print(poetry)
poetry = seed_text_original
seed_text = seed_text_original


print("----------------------------------")
print("Model3")
poetry = seed_text
for ss in range(next_words):
    tokens = tokenizer.tokenize(seed_text)
    res = addLineToDataset(tokens)
    predicted = model3.predict_classes([res], verbose=0)

    word_list = seed_text.split()
    number_of_words = len(word_list)
    if number_of_words >= max_length:
        seed_text = deleteFirstWord(seed_text)

    poetry += " " + tokenizeArray[predicted[0]]
    seed_text = seed_text + " " + tokenizeArray[predicted[0]]

print(poetry)
poetry = seed_text_original
seed_text = seed_text_original


print("----------------------------------")
print("Model4")
poetry = seed_text
for ss in range(next_words):
    tokens = tokenizer.tokenize(seed_text)
    res = addLineToDataset(tokens)
    predicted = model4.predict_classes([res], verbose=0)

    word_list = seed_text.split()
    number_of_words = len(word_list)
    if number_of_words >= max_length:
        seed_text = deleteFirstWord(seed_text)

    poetry += " " + tokenizeArray[predicted[0]]
    seed_text = seed_text + " " + tokenizeArray[predicted[0]]

print(poetry)
poetry = seed_text_original
seed_text = seed_text_original


