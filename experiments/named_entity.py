from . import Experiment
import NER as ner


class NamedEntityExperiment(Experiment):
    def get_query_per_item(self, item):
        return ' '.join(list(ner.get_entities(item['Subject'] + item['Content'])))
