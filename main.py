import json
import argparse
from datetime import datetime
import glob
import os
from tqdm import tqdm

from utils import Utils, bcolors, timing_decorator, calculate_mrr, get_white_listed_ids, length_stats_print
import anserini
from CommandLineActions import CommandLineActions

from experiments.automatic import AutomaticExperminet as AutomaticExp
from experiments.classifier import ClassifierExperminet as ClassifierExp
from experiments.cnn import CNNExperminet as CNNExp

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

    def print_handwritten_stats(self):
        min_len = 1000
        max_len = 0
        sum_len = 0
        for item in self.corpus_gen_white_listed():
            length = len(anserini.tokenizeString(item['Content'] + item['Subject'], 'lucene'))
            max_len = max(max_len, length)
            min_len = min(min_len, length)
            sum_len += length
        print('max q length: {}'.format(max_len))
        print('min q length: {}'.format(min_len))
        print('avg q length: {}'.format(sum_len / 476))

    def corpus_gen_non_white_listed(self):
        for i, item in enumerate(self.data_list):
            if item['Id'] in self.white_list:
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

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))


def main(args):
    corpus = WebisCorpus(args)

    if args.experiment == "stats":
        length_stats_print('queries-silver.p')
        # length_stats_print('id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')
        # corpus.print_handwritten_stats()
        return

    if args.experiment == "cnn":
        CNNExp(corpus, train_samples=100, use_ner=True)
        return


    if args.experiment == 'classifier':
        train_samples = 1000
        ClassifierExp(corpus, train_samples=train_samples, use_ner=True)
        ClassifierExp(corpus, train_samples=train_samples, use_ner=False)
        ClassifierExp(corpus, train_samples=train_samples, use_noun=False)
        ClassifierExp(corpus, train_samples=train_samples, use_verb=False)
        ClassifierExp(corpus, train_samples=train_samples, use_adj=False)
        # ClassifierExp(corpus, train_samples=train_samples, use_ner=True)
        return

    if args.experiment == 'classifier-no-context':
        train_samples = 50
        gain = 50
        best = 0
        while True:
            exp = ClassifierExp(corpus, train_samples=train_samples, use_ner=True, useContext=False)
            score = exp.mrr
            if score > best:
                best = score
                train_samples += gain
            else:
                print('best score was: {} with {} train samples'.format(best, (train_samples-gain)))
                return

    if args.experiment == 'classifier-context':
        expSmallTrain = ClassifierExp(corpus, train_samples=170, noQueryTerms=15, use_ner=True, useContext=True)
        print('best score was: {} with {} train samples'.format(expSmallTrain.mrr, (170)))

        # expTrainALl = ClassifierExp(corpus, train_samples=2000, noQueryTerms=15, use_ner=True, useContext=True)
        # print('best score was: {} with {} train samples'.format(expTrainALl.mrr, (2000)))

        # train_samples = 150
        # best = 0
        # gain = 10
        # while True:
        #     score = exp.mrr
        #     if score > best:
        #         best = score
        #         train_samples += gain
        #     else:
        #         return
        return

    if args.experiment == 'automatic':
        AutomaticExp('automatic', corpus)
        return
    if args.experiment == "tf-idf":
        for k in range(5, 30, 5):
            TfIdfExp('tf-idf-'+str(k), corpus, k)
        return
    if args.experiment == "tf":
        for k in range(5, 30, 5):
            TfExp('tf-'+str(k), corpus, k)
        return
    if args.experiment == "idf":
        for k in range(5, 30, 5):
            IdfExp('idf-'+str(k), corpus, k)
        return
    if args.experiment == "random":
        for k in range(5, 30, 5):
            RandomExp('random-'+str(k), corpus, k)
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
    if args.produce_sheet:
        result_pickle_list = [
                                'result-clf-model-neruse-handwritten-as-goldTrueTrue-nounTrue-verbTrue-adjTrue-476.p'
                                # 'result-clf-model-nerTrue-nounFalse-verbTrue-adjTrue-1000.p.p',
                                # 'result-clf-model-nerFalse-nounTrue-verbTrue-adjTrue-1000.p.p',
                                # 'result-clf-model-nerTrue-nounTrue-verbFalse-adjTrue-1000.p.p',
                                # 'result-clf-model-nerTrue-nounTrue-verbTrue-adjFalse-1000.p.p',
                                # 'result-silver.p',
                                # 'result-handwritten.p',
                                # 'result-automatic.p',
                                # 'result-named-entity-only.p',
                                # 'result-tf-idf-25.p',
                                # 'result-tf-idf-20.p',
                                # 'result-tf-idf-15.p',
                                # 'result-tf-idf-10.p',
                                # 'result-tf-idf-5.p',
                                # 'result-tf-25.p',
                                # 'result-tf-20.p',
                                # 'result-tf-15.p',
                                # 'result-tf-10.p',
                                # 'result-tf-5.p',
                                # 'result-idf-25.p',
                                # 'result-idf-20.p',
                                # 'result-idf-15.p',
                                # 'result-idf-10.p',
                                # 'result-idf-5.p',
                                # 'result-random-25.p',
                                # 'result-random-20.p',
                                # 'result-random-15.p',
                                # 'result-random-10.p',
                                # 'result-random-5.p',
                                # 'azzopardi/result-model-1.p',
                                # 'azzopardi/result-model-2.p',
                                # 'azzopardi/result-model-3.p',
                                # 'azzopardi/result-model-4.p'
                              ]
        with open('output_experiments.txt', 'w') as f:
            for result_pickle in result_pickle_list:
                f.write(','.join(calculate_mrr(result_pickle, get_white_listed_ids())) + '\n')
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
    parser.add_argument('--experiment', '-exp', type=str, required=False,
                        default=None,
                        help='specify experiment to run')
    parser.add_argument('--produce_sheet', '-sheet', action='store_true',
                        help='specify experiment to run')
    main(parser.parse_args())


def get_webis_docs():
    from pyserini import collection, index
    gold_doc_dict = Utils.load_from_pickle('id-gold-doc-dict.p')
    webis_docid_list = gold_doc_dict.values()
    webis_docid_list_first_second_names = [value.split('-')[1] + '-' + value.split('-')[2] for value in gold_doc_dict.values()]
    print(webis_docid_list_first_second_names)
    webis_docid_document_content_dict = {}
    paths_to_traverse = Utils.load_from_pickle('clueweb-paths-to-keep.p')

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
    Utils.dump_to_pickle(webis_docid_document_content_dict, 'cleuweb-webis-id-doc-content-dict' + '.p')
