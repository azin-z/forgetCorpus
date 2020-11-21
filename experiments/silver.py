from . import Experiment
from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids
import anserini as anserini


class SilverExperminet(Experiment):
    def __init__(self, webiscorpus):
        self.gold_doc_content_dict = Utils.load_from_pickle('cleuweb-webis-id-doc-content-dict.p')
        self.additional_stopwords = ('she','this','be','that', 'i\'ve', 'we', 'am','thank', 'you\'ll', 'had', 'does','them', 'him','her', 'can\'t', 'cant', 'were','don\'t', 'out', 'd', 's', 'r', 'help', 'also', 'its', 'his','do', 'his', 'he', 'think', 'has', 'i\'m', 'plz', 'at', 'was', 'thanks', 'or', 'please', 'forgot', 'forgotten', 'remember', 'on', 'the', 'it', 'is', 'a', 'of', 'i', 'what', 'who', 'which', 'so', 'you', 'would', 'me', 'when', 'your', 'can', 'my', 'about', 'from', 'all')
        super(SilverExperminet, self).__init__('silver', webiscorpus)

    def build_queries(self):
        """building dictionaries for queries"""
        id_query_dict = {}
        for item in (self.webiscorpus.corpus_gen()):
            id = item['Id']
            id_query_dict[id] = self.get_query_per_item(item)
        self.dump_query_file(id_query_dict)

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

    def run(self):
        self.build_queries()
        # self.search_queries()
        # _, mrr, _, _ = calculate_mrr(self.result_pickle_name, self.white_list)
        # self.mrr = float(mrr)