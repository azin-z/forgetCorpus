from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
from tqdm import tqdm
import anserini as anserini
from . import Experiment

import operator

class MostCommonTermsExperiment(Experiment):
    def __init__(self, name, webiscorpus):
        self.doc_vector = {}
        self.stopwords = []
        super(MostCommonTermsExperiment, self).__init__(name, webiscorpus)

    # def dump_query_file(self, id_query_dict):
    #     Utils.dump_to_pickle(id_query_dict, self.query_pickle_name)

    # def get_query_per_item(self, item):
    #     return item['Subject'] + item['Content']

    # def build_queries(self):
    #     """building dictionaries for queries"""
    #     id_query_dict = {}
    #     for item in tqdm(self.webiscorpus.corpus_gen_white_listed()):
    #         id = item['Id']
    #         if id in self.white_list:
    #             id_query_dict[id] = self.get_query_per_item(item)
    #     self.dump_query_file(id_query_dict)

    def count_terms_in_item(self, item):
        for i in anserini.tokenizeString(item, 'lucene'):
            if i in self.doc_vector:
                self.doc_vector[i] += 1
            else:
                self.doc_vector[i] = 1

    def get_query_per_item(self, item):
        print(anserini.tokenizeString(item['Subject'] + item['Content'], 'lucene'))
        return ' '.join([item for item in anserini.tokenizeString(item['Subject'] + item['Content'], 'lucene') if item not in anserini.additional_additional_stopword and item not in anserini.stopwords_temp])

    def build_stop_words(self):
        for item in tqdm(self.webiscorpus.corpus_gen()):
            self.count_terms_in_item(item['Subject'] + ' ' + item['Content'])

        sorted_doc_vector = sorted(self.doc_vector.items(), key=operator.itemgetter(1))

        for i, (word, count) in enumerate(sorted_doc_vector[int(0.96 * len(sorted_doc_vector)):]):
            cf = anserini.get_term_coll_freq(word)
            if cf > 250000000 or cf == 0:  # if it's 0 it's stopword
                self.stopwords.append(word)
        print(len(self.stopwords))
        print(self.stopwords)

    def run(self):
        # self.build_stop_words()

        self.build_queries()
        self.search_queries()
        _, mrr, _, _ = calculate_mrr(self.result_pickle_name, self.white_list)
        self.mrr = float(mrr)
