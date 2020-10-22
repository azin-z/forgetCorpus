from . import Experiment
import anserini as anserini


class TfExperiment(Experiment):
    def __init__(self, model_name, webiscorpus, k):
        self.k = k
        super(TfExperiment, self).__init__(model_name, webiscorpus)

    def select_top_words(self, text):
        """selecting top words"""
        terms = list(set(anserini.tokenizeString(text, 'lucene')))
        scores = []
        for i, term in enumerate(terms):
            try:
                tf = anserini.get_term_coll_freq(term)
            except Exception as e:
                print(e)
                continue
            scores.append(tf)

        picked_words = []
        while len(picked_words) < self.k and len(scores):
            picked_word_index = scores.index(min(scores))
            picked_word = terms[picked_word_index]
            scores.pop(picked_word_index)
            terms.pop(picked_word_index)
            if picked_word not in picked_words and \
                            picked_word != 'remember' and picked_word != 'forget':
                picked_words.append(picked_word)
        return ' '.join(picked_words)

    def get_query_per_item(self, item):
        return self.select_top_words(item['Subject'] + item['Content'])
