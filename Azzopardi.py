lambda_val = 0.8
import anserini as anserini
from utils import Utils
import numpy as np
from tqdm import tqdm
from collections import Counter
import time
import math


class AzzopardiFunctions:
    def __init__(self):
        try:
            self.collection_term_count_dict = anserini.Utils.load_from_pickle('clueweb-collection_term_count_dict.p')
        except:
            self.collection_term_count_dict = dump_collection_and_vocab_size
        try:
            self.collection_mini_term_count_dict = anserini.Utils.load_from_pickle('clueweb-mini-collection_term_count_dict.p')
        except:
            self.collection_mini_term_count_dict = make_random_mini_collection_vocab(2)
        self.mini_coll_probs = list(self.collection_mini_term_count_dict.values())
        coll_size = np.sum(self.mini_coll_probs)
        # print('sum for coll size is:', coll_size)
        self.mini_coll_probs = self.mini_coll_probs / coll_size
        self.mini_coll_keys = list(self.collection_mini_term_count_dict.keys())
        self.doc_model_queries = [{} for sub in range(5)]

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
            df = anserini.get_term_doc_freq(term)
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
            Utils.dump_to_pickle(self.doc_model_queries[i], 'azzopardi/queries-model-' + str(i) + '.p')


def make_random_mini_collection_vocab(type):
    #total 689710000
    start_t = time.time()
    print('building mini collection')
    collection_term_count_dict = Utils.load_from_pickle('clueweb-collection_term_count_dict.p')
    print('loaded full collection took', time.time() - start_t)
    mini_collection_term_count_dict = {}
    if type == 1:
        start_t = time.time()
        for i, (term, count) in tqdm(enumerate(collection_term_count_dict.items())):
            if i % 10000 == 0:
                mini_collection_term_count_dict[term] = count
        print('time took to make mini style 1 is', time.time() - start_t)
    Utils.dump_to_pickle(mini_collection_term_count_dict, 'clueweb-mini-collection_term_count_dict-style-1.p')
    mini_collection_term_count_dict_2 = {}

    if type == 2:
        start_t = time.time()
        for i, (term, count) in enumerate(list(collection_term_count_dict.items())[0:689710000:10000]):
            if i % 10000 == 0:
                mini_collection_term_count_dict_2[term] = count
        print('time took to make mini style 2 is', time.time() - start_t)
    Utils.dump_to_pickle(mini_collection_term_count_dict_2, 'clueweb-mini-collection_term_count_dict-style-2.p')
    return mini_collection_term_count_dict_2


def dump_collection_and_vocab_size():
    lambda_val = 0.8
    iterator = anserini.JIndexReaderUtils.getTerms(anserini.reader)
    counter = 0
    collection_size = 0
    coll_term_count_dict = Counter()
    with tqdm(total=689710000) as pbar:
        while iterator.hasNext():
            pbar.update(1)
            counter += 1
            index_term = iterator.next()
            collection_size += index_term.getTotalTF()
            coll_term_count_dict[index_term.getTerm()] = index_term.getTotalTF()
            if counter % 1000000 == 0:
                Utils.dump_to_pickle(collection_size, 'clueweb-collection_vocab_size.p')
                Utils.dump_to_pickle(counter, 'clueweb-collection-size.p')
                Utils.dump_to_pickle(coll_term_count_dict, 'clueweb-collection_term_count_dict.p')
    Utils.dump_to_pickle(collection_size, 'clueweb-collection_vocab_size.p')
    Utils.dump_to_pickle(counter, 'clueweb-collection-size.p')
    Utils.dump_to_pickle(coll_term_count_dict, 'clueweb-collection_term_count_dict.p')
    coll_term_prob_dict = {}
    for term, count in coll_term_count_dict.items():
        coll_term_prob_dict[term] = float(lambda_val * count) / float(collection_size)
    print(len(coll_term_prob_dict))
    Utils.dump_to_pickle(coll_term_prob_dict, 'clueweb-collection_term_prob_dict.p')
    return coll_term_count_dict
