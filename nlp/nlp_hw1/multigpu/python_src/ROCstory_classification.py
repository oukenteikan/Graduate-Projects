
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.

Here is a sample output of a run on a quad-core machine::

  Loading 20 newsgroups dataset for categories:
  ['alt.atheism', 'talk.religion.misc']
  1427 documents
  2 categories

  Performing grid search...
  pipeline: ['vect', 'tfidf', 'clf']
  parameters:
  {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
   'clf__n_iter': (10, 50, 80),
   'clf__penalty': ('l2', 'elasticnet'),
   'tfidf__use_idf': (True, False),
   'vect__max_n': (1, 2),
   'vect__max_df': (0.5, 0.75, 1.0),
   'vect__max_features': (None, 5000, 10000, 50000)}
  done in 1737.030s

  Best score: 0.940
  Best parameters set:
      clf__alpha: 9.9999999999999995e-07
      clf__n_iter: 50
      clf__penalty: 'elasticnet'
      tfidf__use_idf: True
      vect__max_n: 2
      vect__max_df: 0.75
      vect__max_features: 50000

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging
import sys
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#print(__doc__)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=Warning) #DeprecationWarning)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class CustomFeatures(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['sent_len']) #, 'lang_prob'])

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        X_num_token = list()
        #X_count_nouns = list()

        for sentence in x_dataset:
            # takes raw text and calculates type token ratio
            X_num_token.append(len(sentence))

            # takes pos tag text and counts number of noun pos tags (NN, NNS etc.)
            # X_count_nouns.append(count_nouns(sentence))

        X = np.array([X_num_token]).T  #, X_count_nouns]).T

        if not hasattr(self, 'scalar'):
            self.scalar = StandardScaler().fit(X)
        return self.scalar.transform(X)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = np.recarray(shape=(len(posts),),
                               dtype=[('words', object), ('meta', object)])  #('length', object), ('condscore', object), ('score', object), ('normscore', object), ('langpred', bool)])
        for i, text in enumerate(posts):
            elems = text.split('\t')
            words, cs, s, ns, lp = elems[:5]
            #print(elems)
            features['words'][i] = words
            features['meta'][i] = {'length': len(words.split()), 
                'condscore': float(cs), 'score': float(s), 
                'normscore': float(ns), 'langpred': bool(lp)}
            if len(elems) > 5:
                ecs, es, ens, ep = elems[5:]
                features['meta'][i].update({'event_condscore': float(ecs), 
                    'event_score': float(es), 'event_normscore': float(ens), 'event_pred': bool(ep)})
        return features


# #############################################################################
# Load data test
def load_data(filename, suffix):
    contents, labels = [], [] 
    #data = StoryData()
    with open(filename+'.true.'+suffix) as tinf, open(filename+'.false.'+suffix) as finf:
        for line in tinf:
            elems = line.strip()#.split('\t')
            contents.append(elems)
            labels.append(1)
        for line in finf:
            elems = line.strip()#.split('\t')
            contents.append(elems)
            labels.append(0)
    print("data size:", len(contents))
    return [contents, labels] 


def event_orig_mapping(orig_idx_file, event_idx_file):
    orig_idx_array = []
    event_idx_dict = {}
    with open(orig_idx_file) as oinf, open(event_idx_file) as einf:
        oinf.readline()
        einf.readline()
        for line in oinf:
            elems = line.strip().split()
            orig_idx_array.append(elems[0])
        counter = 0
        for line in einf:
            elems = line.strip().split()
            event_idx_dict[elems[0]] = counter
            counter += 1
    origin_to_event = {}
    for i, oidx in enumerate(orig_idx_array):
        if oidx in event_idx_dict:
            origin_to_event[i] = event_idx_dict[oidx]
    print ('map dictionary size:', len(origin_to_event))
    return origin_to_event
            
def add_e2e_scores(original_data_array, event_data_array, origin_to_event):
    assert len(event_data_array) == 2 * len(origin_to_event), (len(event_data_array), len(origin_to_event))
    assert len(original_data_array) >= len(event_data_array)
    half_len = len(original_data_array) / 2
    for i, elems in enumerate(original_data_array):
        if i in origin_to_event:
            original_data_array[i] = elems + '\t' + event_data_array[origin_to_event[i]]
        if i - half_len in origin_to_event:
            #print(i, origin_to_event[i-half_len], len(origin_to_event))
            original_data_array[i] = elems + '\t' + event_data_array[origin_to_event[i-half_len] + len(origin_to_event)]
    return original_data_array

def pairwise_eval(probs):
    mid = int(len(probs) / 2)
    print('middle point: %d' % mid)
    pos = probs[:mid]
    neg = probs[mid:]
    assert len(pos) == len(neg)
    count = 0.0
    for p, n in zip(pos, neg):
        if p[1] > n[1]:
            count += 1.0
    #        print('True')
    #    else:
    #        print('False')
    acc = count/mid
    print('Test result: %.3f' % acc)
    return acc


train_data = load_data(sys.argv[1], sys.argv[3])
test_data = load_data(sys.argv[2], sys.argv[3])

#train_event = load_data(sys.argv[4], sys.argv[6])
#test_event = load_data(sys.argv[5], sys.argv[6])

#train_e2o = event_orig_mapping(sys.argv[7], sys.argv[8])
#test_e2o = event_orig_mapping(sys.argv[9], sys.argv[10])

# add event-to-event info
#train_data[0] = add_e2e_scores(train_data[0], train_event[0], train_e2o)
#test_data[0] = add_e2e_scores(test_data[0], test_event[0], test_e2o)

print('Finished data loading!!')
for elem in train_data[0][:10]:
    print (elem)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('featextract', FeatureExtractor()),

    ('union', FeatureUnion(
      transformer_list=[
        ('meta', Pipeline([
            ('selector', ItemSelector(key='meta')),
            ('vect', DictVectorizer()),
            ('scale', StandardScaler(with_mean=False)),
        ])),
        ('word', Pipeline([
            ('selector', ItemSelector(key='words')),
            ('vect', CountVectorizer(ngram_range=(1,5), max_df=0.9)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('char', Pipeline([
            ('selector', ItemSelector(key='words')),
            ('vect', CountVectorizer(ngram_range=(1,5), analyzer='char', max_df=0.8)),
            ('tfidf', TfidfTransformer()),
        ])),
      ],
      transformer_weights={
          'meta': 0.3,
          'word': 1.0,
          'char': 1.0,
      },
    )),
    ('clf', SGDClassifier(loss='log', alpha=0.0005, tol=0.005, random_state=0)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'union__transformer_weights': ({'meta': 0.6, 'word': 1.0, 'char': 1.0},
#            {'meta': 1.0, 'word': 1.0, 'char': 0.75},
#            {'meta': 1.0, 'word': 1.0, 'char': 0.5},
#            {'meta': 1.0, 'word': 0.75, 'char': 1.0},
#            {'meta': 1.0, 'word': 0.75, 'char': 0.75},
#            {'meta': 1.0, 'word': 0.75, 'char': 0.5},
#            {'meta': 1.0, 'word': 0.5, 'char': 1.0},
#            {'meta': 1.0, 'word': 0.5, 'char': 0.75},
#            {'meta': 1.0, 'word': 0.5, 'char': 0.5},
            {'meta': 0.7, 'word': 1.0, 'char': 1.0},
            {'meta': 0.5, 'word': 1.0, 'char': 1.0},
            {'meta': 0.4, 'word': 1.0, 'char': 1.0},
            {'meta': 0.3, 'word': 1.0, 'char': 1.0},
#            {'meta': 0.75, 'word': 1.0, 'char': 0.75},
#            {'meta': 0.75, 'word': 1.0, 'char': 0.5},
#            {'meta': 0.75, 'word': 0.75, 'char': 1.0},
#            {'meta': 0.75, 'word': 0.75, 'char': 0.75},
#            {'meta': 0.75, 'word': 0.75, 'char': 0.5},
#            {'meta': 0.75, 'word': 0.5, 'char': 1.0},
#            {'meta': 0.75, 'word': 0.5, 'char': 0.75},
#            {'meta': 0.75, 'word': 0.5, 'char': 0.5},
#            {'meta': 0.5, 'word': 1.0, 'char': 1.0},
#            {'meta': 0.5, 'word': 1.0, 'char': 0.75},
#            {'meta': 0.5, 'word': 1.0, 'char': 0.5},
#            {'meta': 0.5, 'word': 0.75, 'char': 1.0},
#            {'meta': 0.5, 'word': 0.75, 'char': 0.75},
#            {'meta': 0.5, 'word': 0.75, 'char': 0.5},
#            {'meta': 0.5, 'word': 0.5, 'char': 1.0},
#            {'meta': 0.5, 'word': 0.5, 'char': 0.75},
#            {'meta': 0.5, 'word': 0.5, 'char': 0.5},
            ),
    'union__word__vect__max_df': (0.7, 0.8, 0.9, 1.0),   #0.5, 
    'union__char__vect__max_df': (0.7, 0.8, 0.9, 1.0),   #0.5, 
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'union__word__vect__ngram_range': ((1, 4), (1, 5)),  # trigram or 5-grams  (1, 4), 
    #'union__char__vect__ngram_range': ((1, 4), (1, 5)),  # trigram or 5-grams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.001, 0.0005, 0.0001),
    #'clf__penalty': ('l2', 'l1'),
    'clf__tol': (5e-3, 1e-3, 5e-4),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    
#    pipeline.fit(train_data[0], train_data[1])
#    probs = pipeline.predict_proba(test_data[0])
#    acc = pairwise_eval(probs)
#    exit(0)
    
    #grid_params = list(ParameterGrid(parameters))
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    #pipeline.fit(train_data[0], train_data[1]) #.contents, train_data.labels)
    '''for params in grid_params:
        print('Current parameters:', params)
        pipeline.set_params(**params)
        pipeline.fit(train_data[0], train_data[1])
        probs = pipeline.predict_proba(test_data[0])
        acc = pairwise_eval(probs)
    exit(0)
    '''
    grid_search.fit(train_data[0], train_data[1])
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print('predicting on the test data...')
    score = grid_search.score(test_data[0], test_data[1])
    print('Test score: %.3f' % score)
    probs = grid_search.predict_proba(test_data[0])
    pairwise_eval(probs)
