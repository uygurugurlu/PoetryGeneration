import tensorflow as tf
import pandas as pd
import numpy as np
import HeceOlcusu
from zemberek import (
    TurkishTokenizer,
)
from zemberek.tokenization.token import Token

model1 = tf.keras.models.load_model('poetryModel20')
model2 = tf.keras.models.load_model('poetryModel40')
model3 = tf.keras.models.load_model('poetryModel60')
model4 = tf.keras.models.load_model('poetryModel80')

modelArray = [model1, model2, model3, model4]

seed_text = "sevmek istiyorum, sevemiyorum"
seed_text_original = seed_text
next_words = 100
max_length = 14
tokenizeArray = pd.read_csv("tokenizeList.csv")
tokenizeArray = tokenizeArray['0'].to_numpy().tolist()

dataset = pd.read_csv("sequence_data.csv").to_numpy()
dataset = np.delete(dataset, 0, axis=1).tolist()

hece_olcusu = 14


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
    return line.partition(' ')[2]


def checkNewLine(word):
    if word == 'newline':
        return False
    return True


sp = seed_text.split()
lastPredicted = [sp[-2], sp[-1]]


def addToLastPredicted(new):
    if lastPredicted[0] == lastPredicted[1] and lastPredicted[0] == new:
        return False
    else:
        lastPredicted[0] = lastPredicted[1]
        lastPredicted[1] = new
    return True


def checkRecurrent(word):
    if word == lastPredicted:
        return False
    return True


def isDeleteNewLine(word):
    if checkNewLine(word):
        return checkRecurrent(word)
    return False


def addToPoetry(poetryStr, new):
    if new == "newline":
        return poetryStr
    else:
        if not addToLastPredicted(new):
            new = HeceOlcusu.getMostSimilarWords(new)
        line = ""
        if (poetryStr[-1] == '\n'):
            line = ""
        else:
            poetryList = poetryStr.split('\n')
            line = poetryList[-1]
        newLine = line + new
        syllableLen = HeceOlcusu.countSyllable(newLine)
        if syllableLen < hece_olcusu:
            return poetryStr + " " + new
        elif syllableLen == hece_olcusu:
            return poetryStr + " " + new + '\n'
        else:
            req = hece_olcusu - HeceOlcusu.countSyllable(line)
            wordsList = HeceOlcusu.getMostSimilarWords(new)
            selected = ""
            for words in wordsList:
                if HeceOlcusu.countSyllable(words[0]) == req:
                    selected = words[0]
                    break
            return poetryStr + " " + selected + '\n'


tokenizer = TurkishTokenizer.builder().accept_all().ignore_types([Token.Type.NewLine,
                                                                  Token.Type.SpaceTab]).build()

modelNum = 1
for model in modelArray:
    print("----------------------------------")
    print("Model", modelNum)
    modelNum += 1
    poetry = seed_text
    for ss in range(next_words):
        tokens = tokenizer.tokenize(seed_text)
        res = addLineToDataset(tokens)
        predicted = model.predict_classes([res], verbose=0)
        if not (isDeleteNewLine(tokenizeArray[predicted[0]])):
            for _ in range(10):
                predicted = model1.predict_classes([res], verbose=0)
                if isDeleteNewLine(tokenizeArray[predicted[0]]):
                    break

        word_list = seed_text.split()
        number_of_words = len(word_list)
        if number_of_words >= max_length:
            seed_text = deleteFirstWord(seed_text)
        poetry = addToPoetry(poetry, tokenizeArray[predicted[0]])
        seed_text = seed_text + " " + tokenizeArray[predicted[0]]

    print(poetry)
    poetry = seed_text_original
    seed_text = seed_text_original
