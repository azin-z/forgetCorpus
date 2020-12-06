from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
from tqdm import tqdm
import anserini as anserini


class Experiment:
    def __init__(self, name, webiscorpus, rm3=False, mini_index=False, rm3Terms=10, rm3Docs=10 , rm3OrigWeight=0.9):
        self.rm3 = rm3
        self.rm3Terms = rm3Terms
        self.rm3Docs = rm3Docs
        self.rm3OrigWeight = rm3OrigWeight
        self.mini_index = mini_index
        self.name = name
        self.webiscorpus = webiscorpus
        self.gold_doc_dict = Utils.load_from_pickle('id-gold-doc-dict.p')
        self.query_pickle_name = 'queries-' + self.name + '.p'
        self.result_pickle_name = 'result-' + self.name
        if self.rm3:
            self.result_pickle_name += '-rm3'
        if self.mini_index:
            self.result_pickle_name += '-mini-index'
        self.result_pickle_name += '.p'
        self.white_list = get_white_listed_ids()
        self.mrr = 0
        self.run()


    def dump_query_file(self, id_query_dict):
        Utils.dump_to_pickle(id_query_dict, self.query_pickle_name)

    def get_query_per_item(self, item):
        return item['Subject'] + item['Content']

    def build_queries(self):
        """building dictionaries for queries"""
        id_query_dict = {}
        for item in tqdm(self.webiscorpus.corpus_gen_white_listed()):
            id = item['Id']
            if id in self.white_list:
                id_query_dict[id] = self.get_query_per_item(item)
        self.dump_query_file(id_query_dict)

    @timing_decorator
    def search_queries(self):
        """Searching clueweb by query file"""
        queries = Utils.load_from_pickle(self.query_pickle_name)
        result_dict = {}
        jqueries = anserini.JList()
        ids = anserini.JList()
        for i, query in enumerate(list(queries.values())):
            jqueries.add(query)
        for i, id in enumerate(list(queries.keys())):
            ids.add(id)

        searcher = anserini.searcher
        if self.mini_index:
            searcher = anserini.minisearcher
        if self.rm3:
            searcher.setRM3Reranker(self.rm3Terms, self.rm3Docs, self.rm3OrigWeight, True)
        results = searcher.batchSearch(jqueries, ids, 500, 40)
        results_key_set = results.keySet().toArray()
        for resultKey in results_key_set:
            for i, result in enumerate(results.get(resultKey)):
                if self.mini_index:
                    if result.docid == resultKey:
                        result_dict[resultKey] = i
                        break
                else:
                    if result.docid == self.gold_doc_dict[resultKey]:
                        result_dict[resultKey] = i
                        break
            if resultKey not in result_dict:
                result_dict[resultKey] = None
        Utils.dump_to_pickle(result_dict, self.result_pickle_name)

    def run(self):
        self.build_queries()
        self.search_queries()
        _, mrr, _, _ = calculate_mrr(self.result_pickle_name, self.white_list)
        self.mrr = float(mrr)
