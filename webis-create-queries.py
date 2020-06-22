import json
import argparse
from datetime import datetime
import glob
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CommandLineActions:
    @staticmethod
    def user_continues():
        return input('Press enter to go to next query. Press anything else to exit. ') == ''

    @staticmethod
    def print_item_info(item):
        print(bcolors.BOLD + 'Subject:\t' + bcolors.ENDC, bcolors.OKGREEN + item['Subject'] + bcolors.ENDC)
        print(bcolors.BOLD + 'Content:\t' + bcolors.ENDC, bcolors.OKGREEN + item['Content'] + bcolors.ENDC)
        print(bcolors.BOLD + 'Chosen answer:\t' + bcolors.ENDC, bcolors.OKGREEN + item['ChosenAnswer'] + bcolors.ENDC)

    @staticmethod
    def modify_existing_query(item):
        if 'ForgetQuery' in item:
            print(bcolors.BOLD + 'Existing query: \t' + bcolors.ENDC, bcolors.OKBLUE + item['ForgetQuery'] + bcolors.ENDC)
            return True
        return False

    @staticmethod
    def log_entered_query(new_query):
        print('Saving following query: ' + bcolors.OKBLUE + new_query + bcolors.ENDC)

    @staticmethod
    def get_new_query(modify=False):
        modify_str = ''
        if modify:
            modify_str = 'new '
        return input('Provide a ' + modify_str + 'query for given item or press enter to skip:\n')


class WebisCorpus:
    def __init__(self, args):
        self.skip_existing = args.skip
        self.num_of_queries = args.num
        self.data_list = None
        self.output_file_name = args.output
        with open(args.corpus) as corpusFile:
            self.data_list = json.loads(corpusFile.read())

    def corpus_gen(self):
        counter = 0
        for i, item in enumerate(self.data_list):
            if counter == self.num_of_queries:
                return
            if self.skip_existing and ('ForgetQuery' in item):
                continue
            counter += 1
            yield item

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))


def main(args):
    corpus = WebisCorpus(args)
    modification_made = False
    for item in corpus.corpus_gen():
        CommandLineActions.print_item_info(item)
        new_query = CommandLineActions.get_new_query(CommandLineActions.modify_existing_query(item))
        if not new_query == '':
            item['ForgetQuery'] = new_query
            CommandLineActions.log_entered_query(new_query)
            modification_made = True
        if not CommandLineActions.user_continues():  # ask if user continues
            break
    if modification_made:
        corpus.write_corpus_to_file()


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
    parser.add_argument('--output', '-o', type=str, nargs=1, required=False, default='corpus/edited-at-' + str(datetime.now()).replace(" ", "_").split(".")[0] + '.json',
                        help='name of file to write the updated corpus to')
    parser.add_argument('--corpus', '-c', type=str, nargs=1, required=False,
                        default=get_most_recent_corpus_file(),
                        help='Name of the corpus file, default value is the most recent file in the corpus folder.')
    main(parser.parse_args())
