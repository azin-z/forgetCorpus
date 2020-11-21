import anserini as anserini
from utils import Utils, timing_decorator
import POS as pos
import NER as ner

import numpy as np

from sklearn import svm


class Classification:
    def __init__(self, use_ner=True, use_noun=True, use_verb=True, use_adj=True, train_samples=1000, useTFIDF=True, useContext=True):
        self.clf = svm.SVC()
        self.use_ner = use_ner
        self.use_verb = use_verb
        self.use_noun = use_noun
        self.use_adj = use_adj
        self.useTFIDF = useTFIDF
        self.useContext = useContext
        self.X = []  # shape (n_samples, n_features)
        self.y = []  # shape n_samples the labels for the samples

    @timing_decorator
    def train(self):
        """training classifier with given examples"""
        self.clf.fit(self.X, self.y)
        self.flush()

    def flush(self):
        self.X = []
        self.y = []

    def predict(self):
        # result = self.clf.predict(self.X)
        result = self.clf.decision_function(self.X)
        self.flush()
        return result

    def save_model(self, name):
        Utils.dump_to_pickle(self.clf, 'classifiermodels/' + name)

    def load_model(self, name):
        self.clf = Utils.load_from_pickle('classifiermodels/' + name)

    def getAccuracy(self, result):
        totalTrue = 0
        totalNotTrue = 0
        correctlyPredictedTrue = 0
        correctlyPredictedNotTrue = 0
        for i in range(len(result)):
            if self.y[i] == 1:
                totalTrue += 1
                if result[i] == 1:
                    correctlyPredictedTrue += 1
            if self.y[i] == 0:
                totalNotTrue += 1
                if self.y[i] == 0:
                    correctlyPredictedNotTrue += 1
        print(len(result))
        print('recall:', correctlyPredictedTrue/totalTrue)
        print('precision:', correctlyPredictedTrue/np.count_nonzero(result == 1))

    def add_sample(self, is_in_doc, features):
        self.X.append(features)
        self.y.append(is_in_doc)

    def process_word(self, i, term, block_type, pos_tag, entity_words, term_doc_count_dict, total_length):
        is_in_subject = int(block_type == 'subject')
        is_in_content = int(block_type == 'content')
        if self.use_noun:
            is_noun = int(pos_tag == 'NOUN')
        else:
            is_noun = 0
        if self.use_verb:
            is_verb = int(pos_tag == 'VERB')
        else:
            is_verb = 0
        if self.use_adj:
            is_adj = int(pos_tag == 'ADJ')
        else:
            is_adj = 0
        if self.use_ner:
            is_entity = int(term in entity_words)
        else:
            is_entity = 0
        tf = float(anserini.get_term_coll_freq(term)) / 689710000
        try:
            idf = 1 / anserini.get_term_doc_freq(term)
        except:
            idf = 0
        tf_in_q = term_doc_count_dict[term] / total_length
        rel_pos = float(i) / total_length
        return [tf, idf, tf_in_q, rel_pos, is_in_subject, is_in_content, is_noun, is_verb, is_adj, is_entity]

    def process_block(self, text, terms, block_type, term_doc_count_dict, total_length, silver_query):
        pos_tags = pos.get_pos_tags(terms)
        entity_words = set()
        if self.use_ner:
            entity_words = ner.get_entities(text)
        size = 10
        prev_prev_features = [0] * size
        prev_features = [0] * size
        next_features = [0] * size
        nex_next_features = [0] * size
        for i, (term, pos_tag) in enumerate(zip(terms, pos_tags)):
            features = self.process_word(i, term, block_type, pos_tag, entity_words, term_doc_count_dict, total_length)
            # if i > 1:
            #     prev_prev_features = self.process_word(i-2, terms[i-2], block_type, pos_tags[i-2], entity_words, term_doc_count_dict, total_length)
            if i > 0:
                prev_features = self.process_word(i-1, terms[i-1], block_type, pos_tags[i-1], entity_words, term_doc_count_dict, total_length)
            if i < len(terms) - 1:
                next_features = self.process_word(i+1, terms[i+1], block_type, pos_tags[i+1], entity_words, term_doc_count_dict, total_length)
            # if i < len(terms) - 2:
            #     nex_next_features = self.process_word(i+2, terms[i+2], block_type, pos_tags[i+2], entity_words, term_doc_count_dict, total_length)
            if self.useContext:
                features = prev_prev_features + prev_features + features + next_features + nex_next_features
            is_in_doc = int(term in anserini.tokenizeString(silver_query, 'lucene'))
            self.add_sample(is_in_doc, features)

    def process_query(self, subject, content, silver_query):
        doc_vector = {}
        for i in anserini.tokenizeString(subject + ' ' + content, 'lucene'):
            if i in doc_vector:
                doc_vector[i] += 1
            else:
                doc_vector[i] = 1
        subject_terms = anserini.tokenizeString(subject, 'lucene')
        content_terms = anserini.tokenizeString(content, 'lucene')
        total_length = len(subject_terms) + len(content_terms)
        self.process_block(subject, subject_terms, 'subject', doc_vector, total_length, silver_query)
        self.process_block(content, content_terms, 'content', doc_vector, total_length, silver_query)
        return subject_terms + content_terms

