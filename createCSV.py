import time
import logging
import pandas as pd
import numpy as np
from zemberek import (
    TurkishTokenizer,
)
from zemberek.tokenization.token import Token

tokenizeArray = []
indexedDataset = []
tokens = []
sequenceArray = []
maxLength = 0


def getMaxLengthFromTokens(tokenarray):
    global maxLength
    global maxIndex
    if len(tokenarray) > maxLength:
        maxLength = len(tokenarray)
        maxIndex = index


def createSequenceArray():
    for i in indexedDataset:
        for j in range(2, len(i) + 1):
            sequenceArray.append(i[0:j])


def addToTokenizeArray(word):
    if word in tokenizeArray:
        return tokenizeArray.index(word)
    else:
        tokenizeArray.append(word)
        return tokenizeArray.index(word)


def addLineToDataset(tokenarray):
    indexed = []
    for i in range(len(tokenarray)):
        indexed.append(addToTokenizeArray(tokenarray[i].content))
    indexed.append(-1)
    indexedDataset.append(indexed)


def applyPadding():
    global sequenceArray
    global maxLength
    newArray = []
    for sa in sequenceArray:
        zeroAmount = maxLength + 1 - len(sa)
        zeroArray = []
        for i in range(zeroAmount):
            zeroArray.append(0)
        concat = zeroArray + sa
        newArray.append(concat)
    sequenceArray = newArray


data = pd.read_csv("poetry_TR.csv")
df = pd.DataFrame(data)
df.columns = ['poetry']
tokenizer = TurkishTokenizer.builder().accept_all().ignore_types([Token.Type.NewLine,
                                                                  Token.Type.SpaceTab, Token.Type.Punctuation]).build()

for index, row in df.iterrows():
    tokens = tokenizer.tokenize(row['poetry'])
    getMaxLengthFromTokens(tokens)
    addLineToDataset(tokens)

createSequenceArray()
applyPadding()
#for x in sequenceArray:
#    print(x)
sequenceArray = pd.DataFrame(data=sequenceArray)
sequenceArray.to_csv("sequence_data.csv")
print("tokenize array length: ", len(tokenizeArray)+1)

tokenizeData = pd.DataFrame(data=tokenizeArray)
tokenizeData.to_csv("tokenizeList.csv")