from . import Experiment
import anserini as anserini


class AutomaticExperminet(Experiment):
    def get_query_per_item(self, item):
        # query = []
        return  ' '.join([item for item in anserini.tokenizeString(item['Subject'] + item['Content'], 'lucene') if item not in anserini.stopwords_temp and item not in anserini.additional_stopwords])
