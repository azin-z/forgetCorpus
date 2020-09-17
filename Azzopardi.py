lambda_val = 0.8
import anserini as anserini
from utils import Utils
import numpy as np

import math
class AzzopardiFunctions:

    def __init__(self):
        self.reader = anserini.JIndexReaderUtils.getReader(anserini.JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
        self.collection_term_count_dict = anserini.Utils.load_from_pickle('clueweb-collection_term_count_dict.p')
        self.collection_mini_term_count_dict = anserini.Utils.load_from_pickle('clueweb-mini-collection_term_count_dict.p')
        self.mini_coll_probs = list(self.collection_mini_term_count_dict.values())
        coll_size = np.sum(self.mini_coll_probs)
        # print('sum for coll size is:', coll_size)
        self.mini_coll_probs = self.mini_coll_probs / coll_size
        self.mini_coll_keys = list(self.collection_mini_term_count_dict.keys())
        self.doc_model_queries = [{} for sub in range(5)]

    def get_term_doc_freq(self, term):
        jterm = anserini.JTerm("contents", term)
        df = self.reader.docFreq(jterm)
        return df

    def build_query_from_probs_dict(self, doc_id, model, query_len, doc_term_prob_dict):
        random_numbers = [np.random.rand() for _ in range(query_len)]
        from_document_count = sum(i <= lambda_val for i in random_numbers)
        probs = list(doc_term_prob_dict.values())
        probs = probs / np.sum(probs)
        use_coll_terms = True
        if from_document_count > len(probs):
            from_document_count = len(probs)
        if query_len - from_document_count < 1:
            use_coll_terms = False

        query_doc = np.random.choice(list(doc_term_prob_dict.keys()), from_document_count, p=probs, replace=False)
        query_coll = ''
        if use_coll_terms:
            query_coll = np.random.choice(self.mini_coll_keys, (query_len - from_document_count), p=self.mini_coll_probs,
                                          replace=False)
        query = ' '.join(query_doc) + ' ' + ' '.join(query_coll)
        print('*** model ' + str(model) + ': ', query)
        self.doc_model_queries[model][doc_id] = query

    def get_document_stats(self, doc_vector):
        doc_term_probs_sum = 0
        sum_doc_term_count = 0
        sum_doc_pop_disc = 0
        doc_term_df = {}
        doc_term_cf = {}
        for term, count in doc_vector.items():
            sum_doc_term_count += count
            if term in anserini.stopwords_temp:
                continue
            cf = self.collection_term_count_dict[term]
            if cf < 1:
                cf = 1
            doc_term_cf[term] = cf
            doc_term_probs_sum += float(1 / cf)
            df = self.get_term_doc_freq(term)
            if df < 1:
                df = 1  # if it's not a stop word it's an oov or sth so 1
            doc_term_df[term] = df
            sum_doc_pop_disc += count * math.log(503892800 / df)
        return {'sum_doc_pop_disc': sum_doc_pop_disc,
                'sum_doc_term_count': sum_doc_term_count,
                'doc_term_df': doc_term_df,
                'doc_term_cf': doc_term_cf,
                'doc_term_probs_sum': doc_term_probs_sum}

    def make_query(self, doc_id, doc_vector, query_len):
        print('in simulate queries')
        collection_size = 267813689169
        doc_term_prob_dict = [{} for sub in range(5)]
        stats = self.get_document_stats(doc_vector)
        for term, count in doc_vector.items():
            if term in anserini.stopwords_temp:
                continue
            cf = stats['doc_term_cf'][term]
            doc_term_prob_dict[1][term] = count / stats['sum_doc_term_count']
            doc_term_prob_dict[2][term] = 1 / len(doc_vector)
            doc_term_prob_dict[3][term] = 1 / float(cf * stats['doc_term_probs_sum'])
            doc_term_prob_dict[4][term] = float(count) * math.log(503892800 / stats['doc_term_df'][term]) / stats['sum_doc_pop_disc']
        for i in range(1, 5):
            self.build_query_from_probs_dict(doc_id, i, query_len, doc_term_prob_dict[i])
            Utils.dump_to_pickle(self.doc_model_queries[i], 'azzopardi/id-pardi-model-' + str(i) + '-with-coll-queries.p')