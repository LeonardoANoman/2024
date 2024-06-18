import string
import re
import string
import re, string
from tqdm.notebook import tqdm
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
def tokenize_twitter(sentences):
    """
    Tokenize sentences into tokens (words)
    
    Args:
        sentences: List of strings
    
    Returns:
        List of lists of tokens
    """
    print("Starting Cleaning Process")
    tokenized_sentences = []
    for sentence in tqdm(sentences):
        sentence = cleanhtml(sentence)
        sentence = _replace_urls(sentence)
        sentence = remove_email(sentence)
        sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        sentence = sentence.lower()
        sentence = misc(sentence)
        tokenized_sentences.append(sentence)
    
    return tokenized_sentences

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def _replace_urls(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    return data

def remove_email(data):
    data = re.sub('\S*@\S*\s?', '', data)
    return data

def misc(data):
    data = re.sub('\s+', ' ', data)
    data = re.sub("\'", "", data)
    data = re.sub("ww+", "", data)
    MAYBE_ROMAN = re.compile(r'(\b[MDCLXVI]+\b)(\.)?', re.I)
    data = re.sub(MAYBE_ROMAN, "", data)
    return data

def littleCleaning(sentences):
    print("Starting cleaning Process")
    ret_list = []
    for sentence in sentences:
      words = sentence.split(" ")
      if len(words) > 5:
        ret_list.append(sentence)
      else:
        continue
    return ret_list

path = '../22/republic.txt'
text = open(path).read().lower()
print('length of the corpus is: :', len(text))

data_list = text.split(".")


pro_sentences = []

def normalization_pipeline(sentences):
    print("Starting Normalization Process")
    sentences = tokenize_twitter(sentences)
    sentences = littleCleaning(sentences)
    print("Normalization Process Finished")
    return sentences

pro_sentences = normalization_pipeline(data_list)
pro_sentences[: 5]

dataText = "".join(pro_sentences[: 700])


def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = clean_doc(dataText)

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
out_filename = '../22/republic_sequences.txt'
save_doc(sequences, out_filename)

in_filename = '../22/republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size=128
epochs=50
model.fit(X, y, batch_size=batch_size, epochs=epochs)

model.save("../22/nextWord.h5")
dump(tokenizer, open('../22/tokenizer.pkl', 'wb'))

import numpy as np

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        predict_x=model.predict(encoded) 
        yhat=np.argmax(predict_x,axis=1)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

in_filename = '../22/republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

print(len(lines))
print(lines[0])

model = load_model("../22/nextWord.h5")

tokenizer = load(open('../22/tokenizer.pkl', 'rb'))

seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

generated = generate_seq(model, tokenizer, seq_length, seed_text, 12)
print(generated)

#