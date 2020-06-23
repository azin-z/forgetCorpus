import json
import argparse
from datetime import datetime
import glob
import os
import time
import functools


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        output = func(*args, **kwargs)
        print(bcolors.WARNING + "{} took {} seconds".format(func.__doc__, round(time.time() - start_time)) + bcolors.ENDC)
        return output
    return wrapper


class CommandLineActions:
    def __init__(self, item):
        self.item = item

    @timing_decorator
    def modify_item(self):
        """Prompting for a single item"""
        self.print_item_info()
        new_query = self.get_new_query(self.modify_existing_query())
        if not new_query == '':
            self.item['ForgetQuery'] = new_query
            self.log_entered_query(new_query)
            return True
        return False

    @staticmethod
    def user_continues():
        return input('Press enter to go to next query. Press anything else to exit. ') == ''

    @staticmethod
    def print_field_title(string):
        print(bcolors.BOLD + string + ':\t', end=bcolors.ENDC)

    @staticmethod
    def print_field_text(string):
        print(bcolors.OKGREEN + string + bcolors.ENDC)

    def print_item_info(self):
        self.print_field_title('Subject')
        self.print_field_text(self.item['Subject'])
        self.print_field_title('Content')
        self.print_field_text(self.item['Content'])
        self.print_field_title('Chosen Answer')
        self.print_field_text(self.item['ChosenAnswer'])

    def modify_existing_query(self):
        if 'ForgetQuery' in self.item:
            self.print_field_title('Existing Query')
            print(bcolors.OKBLUE + self.item['ForgetQuery'] + bcolors.ENDC)
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

    def get_completed_percentage(self):
        done_items = list(x for x in self.data_list if 'ForgetQuery' in x)
        return str(round(len(done_items) / len(self.data_list) * 100, 2)) + '% of items done'

    @timing_decorator
    def get_user_modifications(self):
        """Getting user modifications to corpus"""
        modification_made = False
        for item in self.corpus_gen():
            modification_made = CommandLineActions(item).modify_item() or modification_made
            if not CommandLineActions.user_continues():
                break
        return modification_made

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))


def main(args):
    corpus = WebisCorpus(args)
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
    parser.add_argument('--output', '-o', type=str, nargs=1, required=False, default='corpus/edited-at-' + str(datetime.now()).replace(" ", "_").split(".")[0] + '.json',
                        help='name of file to write the updated corpus to')
    parser.add_argument('--corpus', '-c', type=str, nargs=1, required=False,
                        default=get_most_recent_corpus_file(),
                        help='Name of the corpus file, default value is the most recent file in the corpus folder.')
    main(parser.parse_args())
