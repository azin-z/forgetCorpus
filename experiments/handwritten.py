from . import Experiment


class HandwrittenExperiment(Experiment):
    def get_query_per_item(self, item):
        query = item['ForgetQuery']
        if query == 'NOTFORGETQUERY':
            return ''
        return query
