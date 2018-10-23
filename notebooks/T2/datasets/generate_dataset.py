import re
import pprint
import copy
from collections import namedtuple
import nltk
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

np.warnings.filterwarnings('ignore')

# Download de alguns dataset disponibilizados pelo NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('movie_reviews')
nltk.download('sentence_polarity')
nltk.download('sentiwordnet')
nltk.download('stopwords')
nltk.download('words')

from nltk.corpus import wordnet as wn
from nltk.corpus import movie_reviews
from nltk.corpus import sentiwordnet as wdn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

pp = pprint.PrettyPrinter(indent=4)

neg, pos = movie_reviews.categories()

new_phrases = []
for ids in movie_reviews.fileids(neg):
    for phrase in movie_reviews.sents(ids)[1:]:
        if len(phrase) > 3:
            new_phrases.append({
                'type': 'neg',
                'phrase': ' '.join(phrase).lower(),
                'pos_score': 0.0,
                'neg_score': 0.0,
                'over_score': 0.0
            })
for ids in movie_reviews.fileids(pos):
    for phrase in movie_reviews.sents(ids):
        if len(phrase) > 3:
            new_phrases.append({
                'type': 'pos',
                'phrase': ' '.join(phrase).lower(),
                'pos_score': 0.0,
                'neg_score': 0.0,
                'over_score': 0.0
            })
pp.pprint(new_phrases[:3])

senti_word_net = {}
with open('SentiWordNet_3.0.0_20130122.txt') as fh:
    content = fh.readlines()
    for line in content:
        if not line.startswith('#'):
            data = line.strip().split("\t")
            if len(data) == 6:
                pos_score = float(data[2].strip())
                neg_score = float(data[3].strip())
                if pos_score > 0 or neg_score > 0:
                    pos = data[0].strip()
                    uid = int(data[1].strip())
                    lemmas = [lemma.name() for lemma in wn.synset_from_pos_and_offset(pos, uid).lemmas()]
                    for lemma in lemmas:
                        if lemma in senti_word_net:
                            senti_word_net[lemma]['pos_score'] = pos_score if pos_score > senti_word_net[lemma]['pos_score'] else senti_word_net[lemma]['pos_score']
                            senti_word_net[lemma]['neg_score'] = neg_score if neg_score > senti_word_net[lemma]['neg_score'] else senti_word_net[lemma]['neg_score']
                            senti_word_net[lemma]['obj_score'] = 1 - (senti_word_net[lemma]['pos_score'] + senti_word_net[lemma]['neg_score'])
                        else:
                            senti_word_net[lemma] = {
                                'pos': pos,
                                'id': uid,
                                'pos_score': pos_score,
                                'neg_score': neg_score,
                                'obj_score': 1 - (pos_score + neg_score),
                                'SynsetTerms': [lemma.name() for lemma in wn.synset_from_pos_and_offset(pos, uid).lemmas()]
                            }
print('SentiWordNet size : ', len(senti_word_net))
print('-' * 10)
pp.pprint(next(iter(senti_word_net.items())))

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
transformed_weights = vectorizer.fit_transform([phrase['phrase'] for phrase in new_phrases])
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()

tfidf_word_weights = {}
i = 0
for item in vectorizer.vocabulary_.items():
    tfidf_word_weights[item[0]] = weights[item[1]]
print('TfIdf size : ', len(tfidf_word_weights))
print('-' * 10)
pp.pprint(next(iter(tfidf_word_weights.items())))

n_new_phrases = copy.deepcopy(new_phrases)

wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stwords = set(ENGLISH_STOP_WORDS)

for i, phrase in enumerate(n_new_phrases):
    words = [word for word in phrase['phrase'].split() if len(word) > 1]
    stem_words = [stemmer.stem(word) for word in words]
    lemm_words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    words = [stem if len(stem) > len(lemm_words[i]) else lemm_words[i] for i, stem in enumerate(stem_words)]
    grams = list(ngrams(words, 2, pad_right=True))

    n_grams = []
    for gram in grams:
        v_grams = []
        for word in filter(None, gram):
            word_v = senti_word_net.get(word, None)
            pos_score = 0.0
            neg_score = 0.0
            if word_v:
                pos_score = word_v.get('pos_score')
                neg_score = word_v.get('neg_score')
            v_grams.append((word, pos_score, neg_score))
        n_grams.append(v_grams)
    
    ovr = 0.0
    for n_gram in n_grams:
        g1 = n_gram[0]
        word1, pos1, neg1 = g1
        try:
            g2 = n_gram[1]
            word2, pos2, neg2 = g2
            if pos1 - neg1 >= 0 and pos2 - neg2 >= 0:
                pos_db = 1.0
                if pos1 > 0 and pos2 > 0:
                    pos_db = 1.25
                ovr += ((pos1 - neg1) + (pos2 - neg2)) * pos_db
            elif pos1 - neg1 <= 0 and pos2 - neg2 <= 0:
                neg_db = 1.0
                if neg1 > 0 and neg2 > 0:
                    neg_db = 1.25
                ovr += ((pos1 - neg1) + (pos2 - neg2)) * neg_db
        except IndexError:
            pass

    tfidf = 0.0
    for word in set(words):
        tfidf += tfidf_word_weights.get(word, 0)
    corr = 1 + (tfidf * len(words))
    corr = corr if n_new_phrases[i]['type'] == 'pos' else -corr
    n_new_phrases[i]['over_score'] = corr + ovr

# normalizando os valores
scores = np.array([m['over_score'] for m in n_new_phrases])
a, b, mmin, mmax = -100, 100, np.min(scores), np.max(scores)
gt = np.max([np.abs(mmin), mmax])
mmin = -gt + (-.15)
mmax += .15
scores = np.floor(a + (((scores - mmin) * (b-a)) / (mmax - mmin)))

for i, item in enumerate(n_new_phrases):
    n_new_phrases[i]['over_score'] = scores[i]

print('-' * 20)
print('Frases:')
pp.pprint(n_new_phrases[:5])

with open('movie_review_valence_dataset.txt', 'w') as fhandler:
    for phrase in n_new_phrases:
        fphrase = ''.join(phrase["phrase"])
        fhandler.write(f'{fphrase}|#|{phrase["over_score"]}\n')
