from . import Experiment
from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
from Classification import Classification


class ClassifierExperminet(Experiment):

    def __init__(self, webiscorpus, train_samples=1000, use_ner=True, useTFIDF=True):
        self.train_samples = train_samples
        self.model_name = 'clf-model-ner' + str(use_ner) + '-' + str(self.train_samples) + '.p'
        self.train_samples = train_samples
        self.classification = Classification(use_ner=use_ner, train_samples=train_samples, useTFIDF=useTFIDF)
        self.silver_dict = Utils.load_from_pickle(
            'id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')
        super(ClassifierExperminet, self).__init__(self.model_name, webiscorpus)

    @timing_decorator
    def train_model(self):
        """training classifier"""
        for i, item in enumerate(self.webiscorpus.corpus_gen()):
            if i > self.train_samples:
                break
            self.classification.process_query(item['Subject'], item['Content'], self.silver_dict[item['Id']])
        self.classification.train()
        self.classification.save_model(self.model_name)

    def get_query_per_item(self, item):
        """building dictionaries for item"""
        full_terms = self.classification.process_query(item['Subject'], item['Content'], self.silver_dict[item['Id']])
        result = self.classification.predict().tolist()
        query_terms = []
        while len(query_terms) < 10 and len(result):
            picked_word_index = result.index(max(result))
            picked_word = full_terms[picked_word_index]
            result.pop(picked_word_index)
            full_terms.pop(picked_word_index)
            if picked_word not in query_terms:
                query_terms.append(picked_word)
        # query_terms = [full_terms[i] for i in range(len(result)) if result[i] == 1]
        query_terms = ' '.join(set(query_terms))
        print('classifier:', query_terms)
        print('silver:', self.silver_dict[item['Id']])
        return query_terms
        # else:
        #     result = classification.predict()
        #     classification.getAccuracy(result)

    def run(self):
        try:
            self.classification.load_model(self.model_name)
        except:
            self.train_model()
        self.build_queries()
        self.search_queries()
        calculate_mrr(self.result_pickle_name, self.white_list)
