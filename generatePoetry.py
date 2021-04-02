import tensorflow as tf
import pandas as pd
import numpy as np
import random

from zemberek import (
    TurkishTokenizer,
)
from zemberek.tokenization.token import Token

model = tf.keras.models.load_model('poetryModel2')

seed_text = "bir şarab seli hâlinde dönmenin yezitliği"
next_words = 20
max_length = 11
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
    zeroAmount = max_length + 1 - len(tokenarray)
    zeroArray = []
    for i in range(zeroAmount):
        zeroArray.append(0)
    concat = zeroArray + tokenarray
    return concat


def addLineToDataset(tokenarray):
    indexed = []
    for i in range(len(tokenarray)):
        indexed.append(addToTokenizeArray(tokenarray[i].content))
    indexed.append(-1)

    dataset.append(applyPadding(indexed))
    return applyPadding(indexed)


def addLineToDatasetForPrediction(tokenarray):
    indexed = []
    for i in range(len(tokenarray)):
        indexed.append(addToTokenizeArray(tokenarray[i].content))
    indexed.append(-1)

    dataset.append(applyPadding(indexed))
    return applyPadding(indexed)


def deleteFirstWord(line):
    return (line.partition(' ')[2])


tokenizer = TurkishTokenizer.builder().accept_all().ignore_types([Token.Type.NewLine,
                                                                  Token.Type.SpaceTab, Token.Type.Punctuation]).build()
poetry = seed_text
for _ in range(next_words):
    tokens = tokenizer.tokenize(seed_text)

    res = addLineToDataset(tokens)
    res.pop()
    res = np.array(res)
    predicted = model.predict_classes([[res]], verbose=0)
    if len(seed_text) >= max_length:
        seed_text = deleteFirstWord(seed_text)
    if predicted[0] == 15292 or predicted[0] == 15293:
        seed_text = seed_text + " " + tokenizeArray[random.randint(0, 15291)]
        print("sa")
    else:
        poetry += " " + tokenizeArray[predicted[0]]
        seed_text = seed_text + " " + tokenizeArray[predicted[0]]

print(poetry)
