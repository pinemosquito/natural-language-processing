# fork from https://www.kaggle.com/tunguz/lr-with-words-and-char-n-grams
# get dummies
# word_vectorizaer
# char_vectorizer


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import roc_auc_score

from scipy.sparse import hstack

train = pd.read_csv('../input/train.tsv', delimiter='\t').fillna(' ')
test = pd.read_csv('../input/test.tsv', delimiter='\t').fillna(' ')
sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')

train.head()
test.head()
sampleSubmission.head()
train.Sentiment.unique()

Sentiments = pd.get_dummies(train.Sentiment)
Sentiments.head()

Sentiments[0].values

train_text = train['Phrase']
test_text = test['Phrase']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3),
    max_features=18000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 7),
    max_features=60000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

predictions = []
submission = pd.DataFrame.from_dict({'PhraseId': test['PhraseId']})
for i in range(5):
    train_target = Sentiments[i].values
    classifier = LogisticRegression(C=2.65, solver='sag')

    #cv_prediction = cross_val_predict(classifier, train_features, train_target, cv=3, method='predict_proba')
    #predictions.append(cv_prediction[:, 1])
    #print('CV score for class {} is {}'.format(i, cv_score))

    classifier.fit(train_features, train_target)
    submission[str(i)] = classifier.predict_proba(test_features)[:, 1]


predictions_2 = []
submission_2 = pd.DataFrame.from_dict({'PhraseId': test['PhraseId']})
for i in range(5):
    train_target = Sentiments[i].values
    classifier = LogisticRegression(C=2.6, solver='sag')

    #cv_prediction = cross_val_predict(classifier, train_features, train_target, cv=3, method='predict_proba')
    #predictions_2.append(cv_prediction[:, 1])
    #print('CV score for class {} is {}'.format(i, cv_score))

    classifier.fit(train_features, train_target)
    submission_2[str(i)] = classifier.predict_proba(test_features)[:, 1]


predictions_chars = []
submission_chars = pd.DataFrame.from_dict({'PhraseId': test['PhraseId']})
for i in range(5):
    train_target = Sentiments[i].values
    classifier = LogisticRegression(C=2.6, solver='sag')

    #cv_prediction = cross_val_predict(classifier, train_char_features, train_target, cv=3, method='predict_proba')
    #predictions_chars.append(cv_prediction[:, 1])
    #print('CV score for class {} is {}'.format(i, cv_score))

    classifier.fit(train_char_features, train_target)
    submission_chars[str(i)] = classifier.predict_proba(test_char_features)[:, 1]

