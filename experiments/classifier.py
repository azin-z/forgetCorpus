from . import Experiment
from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
from Classification import Classification


class ClassifierExperminet(Experiment):

    def __init__(self, webiscorpus, train_samples=1000, noQueryTerms=15, use_ner=True, useTFIDF=True, use_noun=True, use_verb=True, use_adj=True, useHandwrittenAsGold=False, useContext=False):
        self.train_samples = train_samples
        self.model_name = 'clf-model-ner' + str(use_ner) + '-use-handwritten-as-gold' + str(useHandwrittenAsGold) + '-useContext' + str(useContext) + '-noun' + str(use_noun) + '-verb' + str(use_verb) + '-adj' + str(use_adj) + '-' + str(self.train_samples) + \
                          '-QueryTerms' + str(noQueryTerms) + '5context'
        self.train_samples = train_samples
        self.classification = Classification(use_ner=use_ner, train_samples=train_samples, use_noun=use_noun, use_verb=use_verb, use_adj=use_adj, useTFIDF=useTFIDF, useContext=useContext)
        self.silver_dict = Utils.load_from_pickle(
            # 'id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')
            'queries-silver.p')
        self.noQueryTerms = noQueryTerms
        self.training_item_generator_func = webiscorpus.corpus_gen_non_white_listed
        if useHandwrittenAsGold:
            self.training_item_generator_func = webiscorpus.corpus_gen_white_listed
            self.silver_dict = Utils.load_from_pickle('queries-handwritten.p')
        super(ClassifierExperminet, self).__init__(self.model_name, webiscorpus)

    @timing_decorator
    def train_model(self):
        """training classifier"""
        for i, item in enumerate(self.training_item_generator_func()):
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
        while len(query_terms) < self.noQueryTerms and len(result):
            picked_word_index = result.index(max(result))
            picked_word = full_terms[picked_word_index]
            result.pop(picked_word_index)
            full_terms.pop(picked_word_index)
            if picked_word not in query_terms:
                query_terms.append(picked_word)
        query_terms = ' '.join(set(query_terms))
        print('classifier:', query_terms)
        print('silver:', self.silver_dict[item['Id']])
        return query_terms

    def run(self):
        # try:
        #     self.classification.load_model(self.model_name)
        # except:
        self.train_model()
        self.build_queries()
        self.search_queries()
        _, mrr, _, _ = calculate_mrr(self.result_pickle_name, self.white_list)
        self.mrr = float(mrr)
