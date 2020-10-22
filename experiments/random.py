from . import Experiment
import anserini as anserini
import random


class RandomExperiment(Experiment):
    def __init__(self, model_name, webiscorpus, k):
        self.k = k
        super(RandomExperiment, self).__init__(model_name, webiscorpus)

    def select_random_words(self, text):
        """selecting top words"""
        terms = list(set(anserini.tokenizeString(text, 'lucene')))
        return ' '.join(random.sample(terms, self.k))

    def get_query_per_item(self, item):
        return self.select_random_words(item['Subject'] + item['Content'])
