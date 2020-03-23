import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.parse.corenlp import CoreNLPParser
import random
from nltk.tokenize import wordpunct_tokenize 

nltk.download('stopwords')
nltk.download('punkt')
raw_dataset_path = './data/formality-corpus/'
scores = []
sentences = []

answers = raw_dataset_path + '/answers'
blog = raw_dataset_path + '/blog'
news = raw_dataset_path + '/news'
email = raw_dataset_path + '/email'
categories = [answers, blog, news, email]

#"""
VC = re.compile('[aeiou]+[^aeiou]+', re.I)
def count_syllables(word):
        return len(VC.findall(word))

def fk_score(num_words, num_syllables, num_sentences):
    return 206.835 -1.015*(float(num_words)/num_sentences) -84.6*(num_syllables/float(num_words))

print("Reading raw dataset files")
for cat in categories:
    print("{:20s}  {:20s}".format("Reading File: ", cat))
    with open(cat, 'r', encoding='utf-8', errors= 'ignore') as f:
        for line in f:
            elements = line.strip().split('\t')
            scores.append(float(elements[0].strip()))
            sentences.append(elements[3].strip())
print("Finished Reading Files")

stopwords = nltk.corpus.stopwords.words('english')

# Length Distribution
formal_lengths = []
informal_lengths = []
for i, sent in enumerate(sentences):
    words = wordpunct_tokenize(sent)
    ch_len = len(sent)
    word_len = len(list(words))
    num_syllables = sum(count_syllables(w) for w in words)
    fk_score_ = fk_score(word_len, num_syllables, 1)
    num_stop = sum([1 for w in words if w.lower() in stopwords])
    caps = sum(map(str.isupper, sent))
    t = (sent, scores[i], ch_len, word_len, fk_score_, num_stop, caps)
    if scores[i] > 0:
        formal_lengths.append(t)
    else:
        informal_lengths.append(t)

formal_ch_len = [item[2] for item in formal_lengths]
informal_ch_len = [item[2] for item in informal_lengths]
formal_word_len = [item[3] for item in formal_lengths]
informal_word_len = [item[3] for item in informal_lengths]
print("Formal Ch Len Stats:")
print("Mean: ", np.mean(formal_ch_len, axis=0))
print("Std: ", np.std(formal_ch_len, axis=0))

print("Informal Ch Len Stats:")
print("Mean: ", np.mean(informal_ch_len, axis=0))
print("Std: ", np.std(informal_ch_len, axis=0))

print("Formal Word Len Stats:")
print("Mean: ", np.mean(formal_word_len, axis=0))
print("Std: ", np.std(formal_word_len, axis=0))

print("Informal Word Len Stats:")
print("Mean: ", np.mean(informal_word_len, axis=0))
print("Std: ", np.std(informal_word_len, axis=0))

# Plot them
sns.distplot([item[2] for item in formal_lengths], color="dodgerblue", label="Formal")
sns.distplot([item[2] for item in informal_lengths], color="red", label="Informal")
plt.legend()
plt.title("Character length")
plt.savefig('ch-len.png')
plt.clf()
sns.distplot([item[3] for item in formal_lengths], color="dodgerblue", label="Formal")
sns.distplot([item[3] for item in informal_lengths], color="red", label="Informal")
plt.legend()
plt.title("Word length")
plt.savefig('word-len.png')
plt.clf()
sns.distplot([item[4] for item in formal_lengths], color="dodgerblue", label="Formal")
sns.distplot([item[4] for item in informal_lengths], color="red", label="Informal")
plt.legend()
plt.title("FK score")
plt.savefig('fk_score.png')
plt.clf()
sns.distplot([item[5] for item in formal_lengths], color="dodgerblue", label="Formal")
sns.distplot([item[5] for item in informal_lengths], color="red", label="Informal")
plt.legend()
plt.title("Stopwords")
plt.savefig('Stopwords.png')

plt.clf()
sns.distplot([item[6] for item in formal_lengths], color="dodgerblue", label="Formal")
sns.distplot([item[6] for item in informal_lengths], color="red", label="Informal")
plt.legend()
plt.title("Caps")
plt.savefig('Caps.png')


# Scikit n-gram model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# OLS, Ridge
from sklearn.linear_model import LinearRegression, Ridge
# model evaluation
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

data = list(zip(scores, sentences))
all_data = formal_lengths + informal_lengths
random.shuffle(all_data)
random.shuffle(data)
#scores, sentences = zip(*data)
sent = [item[0] for item in all_data]
score = [item[1] for item in all_data]
feats = [item[2:] for item in all_data]

"""
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    max_features=100000
    )
"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,3))
sent_vec = vectorizer.fit_transform(sent)
#word_vectorizer.fit(sent)
#sent_vec = word_vectorizer.transform(sent)

scaler = StandardScaler()
feats = scaler.fit_transform(feats)

data_x = np.concatenate([sent_vec.toarray(), feats], axis = 1)

train_x, test_x, train_y, test_y = train_test_split(data_x, score, test_size=0.3, random_state=10)
#test_x, test_y = sentences[:1000], scores[:1000]
#val_x, val_y = sentences[1000:4000], scores[1000:4000]
#train_x, train_y = sentences[4000:], scores[4000:]


for alpha in [0,0.5,1]:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(train_x, train_y)
    sct = ridge_reg.score(train_x, train_y)
    sc = ridge_reg.score(test_x, test_y)
    print("Alpha: ", alpha, "Training Score: ", sct)
    print("Alpha: ", alpha, "Test Score: ", sc)

#"""
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)


