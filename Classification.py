import anserini as anserini
from utils import Utils, timing_decorator
import POS as pos
import NER as ner

import numpy as np

from sklearn import svm


class Classification:
    def __init__(self, use_ner=True):
        self.clf = svm.SVC()
        self.use_ner = use_ner
        self.reader = anserini.JIndexReaderUtils.getReader(
            anserini.JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
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

    def save_model(self, train_samples):
        Utils.dump_to_pickle(self.clf, 'classifiermodels/clf-model-no-common-words-ner-' + str(self.use_ner) + '-' + str(train_samples) + '.p')

    def load_model(self, name):
        self.clf = Utils.load_from_pickle(name)

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

    def get_term_coll_freq(self, term):
        jterm = anserini.JTerm("contents", term.lower())
        cf = self.reader.totalTermFreq(jterm)
        return cf

    def get_term_doc_freq(self, term):
        jterm = anserini.JTerm("contents", term)
        df = self.reader.docFreq(jterm)
        return df

    def add_sample(self, tf, idf, tf_in_q, rel_pos, is_in_subject, is_in_content, is_noun, is_adj, is_entity, is_in_doc):
        self.X.append([tf, idf, tf*idf, tf_in_q, rel_pos, is_in_subject, is_noun, is_adj, is_entity, is_in_content])
        self.y.append(is_in_doc)

    def process_block(self, text, terms, block_type, term_doc_count_dict, total_length, silver_query):
        pos_tags = pos.get_pos_tags(terms)
        entity_words = set()
        if self.use_ner:
            entity_words = ner.get_entities(text)
        for i, (term, pos_tag) in enumerate(zip(terms, pos_tags)):
            is_in_subject = int(block_type == 'subject')
            is_in_content = int(block_type == 'content')
            is_noun = int(pos_tag == 'NOUN')
            is_adj = int(pos_tag == 'ADJ')
            is_entity = int(term in entity_words)
            tf = float(self.get_term_coll_freq(term)) / 689710000
            try:
                idf = 1/self.get_term_doc_freq(term)
            except:
                idf = 0
            tf_in_q = term_doc_count_dict[term] / total_length
            rel_pos = float(i)/total_length
            is_in_doc = int(term in anserini.tokenizeString(silver_query, 'lucene'))
            self.add_sample(tf, idf, tf_in_q, rel_pos, is_in_subject, is_in_content, is_noun, is_adj, is_entity, is_in_doc)

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

