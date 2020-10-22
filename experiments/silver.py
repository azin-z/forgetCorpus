from . import Experiment
from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
import anserini as anserini


class SilverExperminet(Experiment):
    def __init__(self, webiscorpus):
        self.gold_doc_content_dict = Utils.load_from_pickle('cleuweb-webis-id-doc-content-dict.p')
        self.additional_stopwords = ('i', 'what', 'who', 'which', 'so', 'you', 'would', 'me', 'when', 'your', 'can', 'my', 'about', 'from', 'all')
        super(SilverExperminet, self).__init__('silver', webiscorpus)

    def get_query_per_item(self, item):
        itemtext = item['Subject'] + item['Content']
        terms_in_common = []
        terms1 = list(set(anserini.tokenizeString(itemtext, 'lucene')))
        try:
            terms2 = list(set(anserini.tokenizeString(self.gold_doc_content_dict[item['KnownItemId']], 'lucene')))
        except:
            return ''
        for term in terms1:
            if term not in self.additional_stopwords and term not in anserini.stopwords_temp and term in terms2:
                    terms_in_common.append(term)
        return ' '.join(terms_in_common)
