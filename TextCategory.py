from multiprocessing.dummy import Process

from sklearn.utils import shuffle
import docx
import numpy as np
import os
from glob import glob
import re
import pandas as pd
from helper import edit_distance
import deepcut
from pythainlp import word_tokenize
from pythainlp.tokenize import word_tokenize 
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize, isthai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter, defaultdict

class TextPreprocessor(object):
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            print('Creating the object')
            
            cls._instance = super(TextPreprocessor, cls).__new__(cls)
            # Put any initialization here.
            vowels = ('”','“','”','_','!!','–','-','(',')','!','"', "'", '.', '%', '$', '#', '@', '\\', '/', ',', '>', '<', '&')
            stopword_set = frozenset(vowels)
            cls.vowels = vowels
            # cls.TH_stopword = thai_stopwords().union(stopword_set)
            cls.TH_stopword = thai_stopwords()
        return cls._instance

    def tokenize(cls, word):
        tokenize_word = deepcut.tokenize(word)
        tokenize_word = [re.sub(r'\d+', '', word) for word in tokenize_word]
        tokenize_word = [word for word in tokenize_word if len(word.strip()) > 1 and word.strip() not in cls.TH_stopword]
        tokenize_word = [normalize(thai) for thai in tokenize_word if isthai(thai) and (thai not in cls.vowels)]
        if not tokenize_word:
            return ''
        return  ' '.join(tokenize_word)
    
    def word_preprocessing(cls, paragraphs):
        '''
        word pre-processing for new document
        '''
        doc = np.full(len(paragraphs), '', dtype='O')
        print('corpus size:', len(paragraphs))
        for i, paragraph in enumerate(paragraphs):
            # print(f'{i}: {paragraph}')
            try:
                doc[i] = cls.tokenize(paragraph)
            except TypeError:
                print(f'error on index: [{i}], word: {paragraph}')
                continue
        doc = doc[doc != '']
        doc = doc[doc != ' ']
        return doc

class Tfidf:
    '''
    workflow: load_dataset -> init_idf
    '''
    def __init__(self, ngram_range=(1, 1), max_features=3000, min_df_factor=0.01) -> None:
        # self.stop_words = [t.encode('utf-8') for t in list(thai_stopwords())]
        # self.stop_words = frozenset(thai_stopwords())
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df_factor = min_df_factor
        self.datasets = None
        self.p = TextPreprocessor()
            
    @staticmethod
    def word_preprocessing(paragraphs):
        p = TextPreprocessor()
        doc = p.word_preprocessing(paragraphs)
        return doc
    
    def load_dataset(self, fname, load_size=None, min_length=80, replace=False):
        # get file format
        file_format = fname.split('.')[-1].lower()
        if file_format not in ['csv']:
            raise NotImplementedError(f'format [{file_format}]: not support')
        # load documet
        df = pd.read_csv(fname, sep='|')
        df = df.dropna(axis=0)
        df = df[df['Summary'].str.len() >= min_length]
        if load_size is None:
            paragraphs = df['Summary'].tolist()
        else:
            paragraphs = df['Summary'].tolist()[:load_size]
        # initial array of string object (corpus) assume 1 paragraph is document
        doc = self.p.word_preprocessing(paragraphs)
        if (self.datasets is None) or (replace is True):
            self.datasets = doc
        else:
            self.datasets = np.hstack([self.datasets, doc])

    def split_tokenizer(self, s):
        return re.split(r'\s+', s)

    def init_idf(self):
        if self.datasets is None:
            raise NotInitializedValue('please call load_dataset')
        # initial tfidf
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.split_tokenizer,
            ngram_range=self.ngram_range,
            # stop_words=self.stop_words,
            # min_df=int(self.datasets.shape[0] * self.min_df_factor),
            # min_df= 2 ,
            max_features=self.max_features,
        )
        # fit the dataset
        self.fit(self.datasets)
        self.tfidf_vectorizer.vocabulary = list(self.tfidf_vectorizer.vocabulary_.keys())
        
    def get_vectorizer(self):
        return self.tfidf_vectorizer
    
    def fit(self, corpus):
        # overide method fit on TfidfVectorizer
        self.tfidf_vectorizer.fit(corpus)
        
    def transform(self, corpus):
        # overide method transform on TfidfVectorizer
        return self.tfidf_vectorizer.transform(corpus)
        
    def fit_transform(self, corpus):
        # overide method fit_transform on TfidfVectorizer
        self.fit(corpus)
        return self.transform(corpus)
    
    def get_feature_names(self):
        # overide method get_feature_names on TfidfVectorizer
        return np.array(self.tfidf_vectorizer.get_feature_names())
    
    def idf_(self, corpus=None, topn=20):
        """get idf word and score

        Args:
            corpus ([None], np.ndarray): corpus of document. Defaults to None.
            topn (int): maximum idf sequence. Defaults to 20.

        Returns:
            dict: word is key and idf score is value
        """        
        if corpus is None:
            corpus = self.datasets
        corpus_idf = self.transform(corpus).toarray()
        corpus_idf = corpus_idf.sum(axis=0)
        rank_idx = np.argsort(corpus_idf)[::-1]
        return dict(zip(self.get_feature_names()[rank_idx[:topn]], corpus_idf[rank_idx[:topn]]))

class TextCategory:
    def __init__(self, idf=None) -> None:
        if idf is None:
            idf = Tfidf()
            idf.load_dataset('dataLDA_01.csv')
            idf.init_idf()
        self.vectorizer = idf
        idf_transform = idf.transform(idf.datasets)
        # create LDA for every tag on dataset
        self.LDA = LDA(n_components=5, learning_decay=0.5, n_jobs=-1, random_state=1234)
        self.LDA.fit(idf_transform)
        # self.unique_keyword = pd.read_csv('unique_keyword.csv')['unique'].to_numpy()
        
        df = pd.read_csv('keyword_new.csv', sep='|')
        self.title = df['title'].to_numpy()
        del df
        with open('unique_keyword_idx_new.txt', encoding='utf-8') as file:
            self.unique_keyword_idx = eval(file.read())
            self.unique_keyword = list(self.unique_keyword_idx.keys())
            self.unique_keyword = np.array(self.unique_keyword)
        
    def fit_with_tag(self, corpus):
        idf_transform = self.vectorizer.fit_transform(corpus)
        self.LDA.fit(idf_transform)
        
    def transform_with_tag(self, corpus):
        idf_transform = self.vectorizer.transform(corpus)
        return self.LDA.transform(idf_transform)
    
    def fit_transform_with_tag(self, corpus):
        idf_transform = self.vectorizer.fit_transform(corpus)
        return self.LDA.fit_transform(idf_transform)
    
    def get_topics(self, corpus, topn=10):
        """get LDA topic from unknown tag with probability based on Tfidf
        Args:
            corpus (np.ndarray of unicode): corpus of document
            topn (int): maximum topics sequence. Defaults to 10.

        Returns:
            [np.ndarray]: topics 
        """        
        corpus = [*filter(lambda c: c != '', corpus)]
        idf_transform = self.vectorizer.fit_transform(corpus)
        self.LDA.fit_transform(idf_transform)
        # print(self.LDA.score(idf_transform))
        topics_feature = self.vectorizer.get_feature_names()
        top_topics_idx = np.fliplr(np.argsort(self.LDA.components_))[..., :topn]
        lda = []
        topics = []
        prob = np.zeros_like(top_topics_idx, dtype=np.float32)
        for i_prob, (values, idx) in enumerate(zip(self.LDA.components_, top_topics_idx)):
            tmp = []
            t = []
            # sum_prob = np.sum(values[idx])
            for j_prob, (i, s) in enumerate(zip(idx, topics_feature[idx])):
                s = re.sub(r'\s+', '', s)
                prob[i_prob, j_prob] = values[i]
                t.append(s)
                tmp.append(f'{s} [{values[i]:.3f}]')
            lda.append(', '.join(tmp))
            topics.append(t)
        print(topics)
        sum_prob = np.sum(prob, axis=1)
        print(prob, sum_prob)
        topics = topics[np.argmax(sum_prob)]
        print(topics)
            # lda.append(dict(zip(topics_feature[idx], values[idx])))
        print('--------------------'*5)
        # lda = list(zip(topics_feature[top_topics_idx], self.LDA[tag_name].components_[..., top_topics_idx]))
        # for (key, value) in lda:
        topics_in_keyword = []
        for t in topics:
            isin_unique = np.vectorize(lambda s: t[:3] in s)
            word_idx = isin_unique(self.unique_keyword)
            word = self.unique_keyword[word_idx]
            if word.shape[0] != 0:
                if t in word:
                    topics_in_keyword.append(t)
                else:
                    distance = edit_distance(word, t)
                    idx = np.argmin(distance)
                    topics_in_keyword.append(word[idx])
                    # print(distance)
            else:
                distance = edit_distance(self.unique_keyword, t)
                # print(distance)
                idx = np.argmin(distance)
                topics_in_keyword.append(self.unique_keyword[idx])
        # print('--------------------'*10)
        # print(lda)
        print('------------'*5)
        print(topics_in_keyword)
        topics_idx = []
        counter = defaultdict(int)
        for tk in topics_in_keyword:
            for idx in self.unique_keyword_idx[tk]:
                counter[idx] += 1
        counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
        candidate = []
        for i, (k, v) in enumerate(counter.items()):
            if i == 0:
                max_value = v
            if max_value == v:
                candidate.append(k)
            else:
                break
        print(candidate)
        for c in candidate:
            print(c, counter[c], self.title[c])
        return lda, self.title[candidate], self.title[np.random.choice(candidate, 1)[0]]
        # topics = remove_space_func(topics_feature[top_topics_idx])
        # return self.get_topics_with_tag(tag_name, corpus, topn)

class NotInitializedValue(Exception):
    pass


if __name__ == '__main__':
    
    size = 100
    topn = 15
    min_length = 80
    print('write LDA')
    fname = 'dataLDA_01.csv'
    idf = Tfidf()
    idf.load_dataset(fname)
    # idf.load_dataset(fname, size)
    idf.init_idf()
    idf_transform = idf.transform(idf.datasets)
    # create LDA for every tag on dataset
    lda = LDA(n_components=5, learning_decay=0.5, n_jobs=-1, random_state=1234)
    lda.fit(idf_transform)
    df = pd.read_csv(fname, sep='|')
    df = df.dropna(axis=0)
    col = list(df.columns)
    data = {'title': [], 'word': []}
    # print(idf.tfidf_vectorizer.vocabulary)
    import pickle
    try:
        with open('idf.pkl', 'wb') as file:
            pickle.dump(idf, file)
    except:
        print(f'dump idf pickle error')
    try:
        with open('lda.pkl', 'wb') as file:
            pickle.dump(lda, file)
    except:
        print(f'dump lda pickle error')
    print('write keyword')
    for i, record in enumerate(df.to_records(index=False)):
        print(i)
        try:
            record = dict(zip(col.copy(), record))
            if len(record['Summary']) < min_length:
                continue
            corpus = idf.p.word_preprocessing([record['Summary']])
            idf_transform = idf.fit_transform(corpus)
            lda.fit_transform(idf_transform)
            topics_feature = idf.get_feature_names()
            top_topics_idx = np.fliplr(np.argsort(lda.components_))[..., :topn]
            lda_topic = []
            topics = []
            prob = np.zeros_like(top_topics_idx)
            for i_prob, (values, idx) in enumerate(zip(lda.components_, top_topics_idx)):
                tmp = []
                t = []
                for j_prob, (i, s) in enumerate(zip(idx, topics_feature[idx])):
                    s = re.sub(r'\s+', '', s)
                    prob[i_prob, j_prob] = values[i]
                    t.append(s)
                    tmp.append(f'{s} [{values[i]:.3f}]')
                lda_topic.append(', '.join(tmp))
                topics.append(t)
            # print(topics)
            sum_prob = np.sum(prob, axis=1)
            topics = topics[np.argmax(sum_prob)]
            data['title'].append(record['Title'])
            data['word'].append('/sp/'.join(topics))
        except:
            print(f'error on index:', i)
    df = pd.DataFrame(data)
    df.to_csv('keyword_new.csv', sep='|', index=False)
        # print(topics)
        # print('--------'*10)

        # record['Title']
        # record['Summary']