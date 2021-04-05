# Python program to generate word vectors using Word2Vec
import pandas as pd
# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

from zemberek import (
    TurkishTokenizer,
)
from zemberek.tokenization.token import Token
import gensim
from gensim.models import Word2Vec

data = pd.read_csv("poetry_TR.csv")
df = pd.DataFrame(data)
df.columns = ['poetry']

tokenizer = TurkishTokenizer.builder().accept_all().ignore_types([Token.Type.NewLine,
                                                                  Token.Type.SpaceTab]).build()
data = [[]]

for index, row in df.iterrows():
    tokens = tokenizer.tokenize(row['poetry'])
    for t in tokens:
        data[0].append(t.content)

# Replaces escape character with space

# iterate through each sentence in the file


# Create CBOW model
model1 = gensim.models.Word2Vec(sentences=data, min_count=1, vector_size=100, window=5, epochs=20)


def getMostSimilarWords(word):
    return model1.wv.most_similar(word, topn=1000)


def getMostSimilarWord(word):
    return model1.wv.most_similar(word, topn=1)


def countSyllable(sentence):
    unlu = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü', 'â']
    syllables = 0
    for i in sentence:
        if (i in unlu):
            syllables += 1

    return syllables
