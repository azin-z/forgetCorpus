from . import Experiment
from utils import Utils, bcolors, timing_decorator, calculate_mrr, get_white_listed_ids, is_clueweb_id, get_webis_id
from Azzopardi import AzzopardiFunctions
from tqdm import tqdm
import anserini as anserini


class AzzopardiExperiment(Experiment):
    def __init__(self, webiscorpus, search_only=True):
        self.search_only = search_only
        super(AzzopardiExperiment, self).__init__('azzopardi', webiscorpus)

    @timing_decorator
    def make_whitelisted_queries(self):
        for i in range(1, 5):
            a = Utils.load_from_pickle('azzopardi/queries-model-' + str(i) + '.p')
            b = {}
            for key, value in a.items():
                if is_clueweb_id(key):
                    key = get_webis_id(key)
                if key in self.white_list:
                    b[key] = value
            Utils.dump_to_pickle(b, 'azzopardi/whiteliste-queries-model-' + str(i) + '.p')

    def run(self):
        self.make_whitelisted_queries()
        if not self.search_only:
            self.build_queries()
        for i in range(1, 5):
            self.query_pickle_name = 'azzopardi/whiteliste-queries-model-' + str(i) + '.p'
            self.result_pickle_name = 'azzopardi/result-model-' + str(i) + '.p'
            self.search_queries()
            calculate_mrr(self.result_pickle_name, self.white_list)

    def build_queries(self):
        id_doc_text = Utils.load_from_pickle('cleuweb-webis-id-doc-content-dict.p')
        azzopardifuncs = AzzopardiFunctions()
        for id in tqdm(id_doc_text.keys()):
            doc_vector = {}
            for i in anserini.tokenizeString(id_doc_text[id], 'lucene'):
                if i in doc_vector:
                    doc_vector[i] += 1
                else:
                    doc_vector[i] = 1
            print(self.get_doc_url(id))
            try:
                azzopardifuncs.make_query(id, doc_vector, 10)
            except Exception as e:
                print('error ', e, 'occured in processing', id)

