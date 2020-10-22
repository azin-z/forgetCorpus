import json
import argparse
from datetime import datetime
import glob
import os
import pickle
from tqdm import tqdm

from utils import Utils, bcolors, timing_decorator, calculate_mrr, get_white_listed_ids
import anserini as anserini
from CommandLineActions import CommandLineActions

from experiments.automatic import AutomaticExperminet as AutomaticExp
from experiments.classifier import ClassifierExperminet as ClassifierExp
from experiments.tfidf import TfIdfExperiment as TfIdfExp
from experiments.tf import TfExperiment as TfExp
from experiments.idf import IdfExperiment as IdfExp
from experiments.random import RandomExperiment as RandomExp
from experiments.handwritten import HandwrittenExperiment as HandwrittenExp
from experiments.named_entity import NamedEntityExperiment as NamedEntityExp
from experiments.azzopardi import AzzopardiExperiment as AzzopardiExp
from experiments.silver import SilverExperminet as SilverExp


class WebisCorpus:
    def __init__(self, args):
        self.skip_existing = args.skip
        self.num_of_queries = args.num
        self.data_list = None
        self.output_file_name = args.output
        self.showAnswer = args.showAnswer
        self.white_list = get_white_listed_ids()
        with open(args.corpus) as corpusFile:
            self.data_list = json.loads(corpusFile.read())

    def get_doc_url(self, id):
        for i, item in enumerate(self.data_list):
            if item['KnownItemId'] == id:
                return item['KnownItemUrl']
        return None

    def get_doc_id(self, cluewebid):
        for i, item in enumerate(self.data_list):
            if item['KnownItemId'] == cluewebid:
                return item['Id']
        return None

    def corpus_gen(self):
        counter = 0
        for i, item in enumerate(self.data_list):
            if counter == self.num_of_queries:
                return
            if self.skip_existing and ('ForgetQuery' in item):
                continue
            counter += 1
            yield item

    def corpus_gen_white_listed(self):
        counter = 0
        for i, item in enumerate(self.data_list):
            if counter == self.num_of_queries:
                return
            if self.skip_existing and ('ForgetQuery' in item):
                continue
            counter += 1
            if item['Id'] not in self.white_list:
                continue
            yield item

    def get_completed_percentage(self):
        done_count = len(list(x for x in self.data_list if 'ForgetQuery' in x))
        total_count = len(self.data_list)
        done_percentage = round(done_count/total_count*100, 2)
        return str(done_percentage) + '% of items done'

    @timing_decorator
    def get_user_modifications(self):
        """Getting user modifications to corpus"""
        modification_made = False
        for item in self.corpus_gen():
            modification_made = CommandLineActions(item, self.showAnswer).modify_item() or modification_made
            if not CommandLineActions.user_continues():
                break
        return modification_made

    def categories(self):
        category_ids = (396546048, 396546049, 396546050, 396546058, 396546061, 396546062, 396546088, 396545576, 396545581, 396545584, 396545591, 396545598, 396547134, 396546114, 396545607, 396545610, 396547158, 396545532, 396547166, 2115500137, 2115500139, 2115500141, 396545134, 396545136, 396545137, 396545138, 396545650, 396545653, 2115500152, 396545663, 396545664, 396545665, 2115500161, 396545160, 396545162, 396545163, 2115500179, 2115500180, 2115500182, 2115500183, 2115500184, 2115500185, 396545183, 396545187, 396545191, 2115500200, 2115500201, 2115500202, 2115500203, 396545196, 2115500204, 396545198, 2115500207, 2115500206, 2115500209, 2115500205, 2115500208, 396545211, 396545212, 396545214, 396545217, 396545219, 396545223, 396545231, 396545232, 396545233, 396545234, 396546301, 396545298, 396545299, 396545300, 396545302, 396545305, 396545307, 396545308, 396545310, 396545316, 396545320, 396545322, 396545324, 396546352, 396545342, 396545359, 396546385, 396545363, 396545364, 396545365, 396545366, 396545368, 396545372, 396545374, 396545377, 396545380, 396546406, 396545382, 396545383, 396546411, 396546415, 396545392, 396546419, 396545397, 396546945, 396546440, 396546444, 396545440, 396545441, 396545442, 396545447, 396545449, 396545453, 396546992, 396545457, 396545458, 396545462, 396545473, 396546501, 396546502, 396547017, 396545488, 396545489, 396545493, 396546519, 396545496, 396546218, 396545504, 396547043, 396546021, 396546035, 396546036, 396546037, 396546039, 396546041, 396546043, 396546044)
        id_category_dict = {}
        for id in category_ids:
            id_category_dict[id] = {}
        max_len = 0
        id_doc_text = Utils.load_from_pickle('cleuweb-webis-id-doc-content-dict.p')

        for item in tqdm(self.corpus_gen()):
            # .push(item['Id'])
            doc_vector = {}
            for i in anserini.tokenizeString(id_doc_text[id], 'lucene'):
                if i in doc_vector:
                    doc_vector[i] += 1
                else:
                    doc_vector[i] = 1
            id_category_dict[item['CategoryId']] = doc_vector
        Utils.dump_to_pickle(id_category_dict, 'categories/id_category_dict.p')

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))


def get_webis_docs():
    from pyserini import collection, index
    gold_doc_dict = pickle.load(open('id-gold-doc-dict.p', 'rb'))
    webis_docid_list = gold_doc_dict.values()
    webis_docid_list_first_second_names = [value.split('-')[1] + '-' + value.split('-')[2] for value in gold_doc_dict.values()]
    print(webis_docid_list_first_second_names)
    # webis_docid_list_first_names = [value.split('-')[1] for value in gold_doc_dict.values()]
    webis_docid_document_content_dict = {}
    paths_to_traverse = pickle.load(open('clueweb-paths-to-keep.p', 'rb'))

    for path in tqdm(paths_to_traverse):
        c = collection.Collection('ClueWeb09Collection', '/GW/D5data/Clueweb09-english/'+path)
        generator = index.Generator('DefaultLuceneDocumentGenerator')
        for (i, fs) in enumerate(c):
            secondName = fs.segment_name.split('.')[0]  # this is the 03 thing
            firstName = path.split('/')[1]  # this is the en000 thing
            name = firstName + '-' + secondName
            print(name)
            if not name in webis_docid_list_first_second_names:
                continue
            print('iterating', firstName + '-' + secondName, 'segment')
            for (j, doc) in tqdm(enumerate(fs)):
                try:
                    parsed = generator.create_document(doc)
                    docid = parsed.get('id')  # FIELD_ID
                    if docid in webis_docid_list:
                        webis_docid_document_content_dict[docid] = parsed.get('contents')
                except:
                    print('null')
    pickle.dump(webis_docid_document_content_dict, open('cleuweb-webis-id-doc-content-dict' + '.p', "wb"))



def main(args):
    corpus = WebisCorpus(args)
    if args.experiment == 'classifier':
        train_samples = 1000
        ClassifierExp(corpus, train_samples=train_samples, use_ner=(not args.no_ner))
        return
    if args.experiment == 'automatic':
        AutomaticExp('automatic', corpus)
        return
    if args.experiment == "tf-idf-queries":
        for k in range(5, 30, 5):
            TfIdfExp('tf-idf-'+str(k), corpus, k)
        return
    if args.experiment == "tf-queries":
        for k in range(5, 30, 5):
            TfExp('tf-'+str(k), corpus, k)
        return
    if args.experiment == "idf-queries":
        for k in range(5, 30, 5):
            IdfExp('idf-'+str(k), corpus, k)
        return
    if args.experiment == "random-queries":
        for k in range(5, 30, 5):
            RandomExp('idf-'+str(k), corpus, k)
        return
    if args.experiment == "handwritten":
        HandwrittenExp('handwritten', corpus)
        return
    if args.experiment == 'ner-only':
        NamedEntityExp('named-entity-only', corpus)
        return
    if args.experiment == 'azzopardi':
        AzzopardiExp(corpus, search_only=True)
        return
    if args.experiment == 'silver':
        SilverExp(corpus)
        return
    if args.experiment == 'make-experiment-sheet':
        result_pickle_list = [
                                'result-silver.p',
                                'result-handwritten.p',
                                'result-automatic.p',
                                'result-named-entity-only.p',
                                'result-tf-idf-25.p',
                                'result-tf-idf-20.p',
                                'result-tf-idf-15.p',
                                'result-tf-idf-10.p',
                                'result-tf-idf-5.p',
                                'result-tf-25.p',
                                'result-tf-20.p',
                                'result-tf-15.p',
                                'result-tf-10.p',
                                'result-tf-5.p',
                                'result-idf-25.p',
                                'result-idf-20.p',
                                'result-idf-15.p',
                                'result-idf-10.p',
                                'result-idf-5.p',
                                'result-random-25.p',
                                'result-random-20.p',
                                'result-random-15.p',
                                'result-random-10.p',
                                'result-random-5.p',
                                'azzopardi/result-model-1.p',
                                'azzopardi/result-model-2.p',
                                'azzopardi/result-model-3.p',
                                'azzopardi/result-model-4.p'
                              ]
        with open('output_experiments.txt', 'w') as f:
            for result_pickle in result_pickle_list:
                f.write(','.join(calculate_mrr(result_pickle, get_white_listed_ids())) + '\n')
        return
    if args.search:
        print('search')
        if args.resultpickle:
            calculate_mrr(args.resultpickle)
        return
    if corpus.get_user_modifications():
        corpus.write_corpus_to_file()
    print(corpus.get_completed_percentage())


if __name__ == "__main__":

    def get_most_recent_corpus_file():
        list_of_files = glob.glob('corpus/*.json')
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    parser = argparse.ArgumentParser(description='Create forget queries for Webis corpus data')
    parser.add_argument('--num', '-n', type=int,  nargs=1, required=False, default=2755,
                        help='Program will stop after n queries are entered')
    parser.add_argument('--skip', '-s', type=bool, nargs='?', const=True, required=False, default=False,
                        help='skip over items with an already existing forget query')
    parser.add_argument('--output', '-o', type=str, required=False, default='corpus/edited-at-' + str(datetime.now()).replace(" ", "_").split(".")[0] + '.json',
                        help='name of file to write the updated corpus to')
    parser.add_argument('--corpus', '-c', type=str, required=False,
                        default=get_most_recent_corpus_file(),
                        help='Name of the corpus file, default value is the most recent file in the corpus folder.')
    parser.add_argument('--showAnswer', '-sa', type=bool, nargs='?',  const=True, required=False,
                        default=False,
                        help='show the chosen answer for the question when prompting')
    parser.add_argument('--search', '-se', type=bool, nargs='?', const=True, required=False,
                        default=False,
                        help='search corpus with specified queries')
    parser.add_argument('--resultpickle', '-rpickle', type=str, required=False,
                        default=False,
                        help='result pickle file to calculate mrr for')
    parser.add_argument('--experiment', '-exp', type=str, required=False,
                        default=None,
                        help='specify experiment to run')
    parser.add_argument('--index', '-ind', type=str, required=False,
                        default='full',
                        help='specify index to use, mini(only webis docs) or full(all of clueweb09) ')
    parser.add_argument('--no_ner', '-no_ner', type=bool, required=False, nargs='?', const=True,
                        default=False,
                        help='use named entities for classifier')
    main(parser.parse_args())
