import json
import getopt, sys
from datetime import datetime


class inputString:
    def __init__(self, string):
        self.string = string

    def is_true(self):
        return self.string == 'y'

    def is_false(self):
        return self.string == 'n'

    def is_valid_query(self):
        return self.string == 'n'


class WebisCorpus:

    def __init__(self, path, skip_existing, num_of_queries,
                 output_file_name):
        self.skip_existing = skip_existing
        self.num_of_queries = num_of_queries
        self.data_list = None
        self.output_file_name = output_file_name
        with open(path) as corpusFile:
            self.data_list = json.loads(corpusFile.read())

    def corpus_gen(self):
        counter = 0
        for item in self.data_list:
            if counter == self.num_of_queries:
                return
            if self.skip_existing and ('ForgetQuery' in item):
                continue
            yield item
            counter += 1

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))


def main(arguments):
    skip_existing = False
    num_of_queries = 2755
    output_file = 'edited-webis-' + str(datetime.now()) + '.json'

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("Displaying help")
            print("Put -s to skip over items with an already existing forget query")
            print("Put -n to specify how many queries you would like to write")
        elif current_argument in ("-o", "--output"):
            print("Changed corpus will be saved to(%s)" % current_value)
            output_file = current_value
        elif current_argument in ("-s", "--skipExisting"):
            print("Will be skipping over items that already have a forget query")
            skip_existing = True
        elif current_argument in ("-n", "--numberOfQueries"):
            print("Program will stop after you've entered (%s) queries" % current_value)
            num_of_queries = int(current_value)

    webis_corpus = WebisCorpus('corpus-webis-kiqc-13.json', skip_existing, num_of_queries, output_file)
    for i, item in enumerate(webis_corpus.corpus_gen()):
        if i > 0:
            user_continues = inputString(input('Go to next query?(y/n)')).is_true()
            if not user_continues:
                break
        print('Subject:\t\t', item['Subject'])
        print('Content:\t\t', item['Content'])
        print('Chosen answer:\t\t', item['ChosenAnswer'])
        if 'ForgetQuery' in item:
            print('***Following query exists for question: \t', item['ForgetQuery'])
            user_inputs_new_query = inputString(input('Would you like to change it?(y/n)')).is_true()
            if not user_inputs_new_query:
                continue
        item['ForgetQuery'] = input('Provide a query for given item:\n')
    webis_corpus.write_corpus_to_file()


if __name__ == "__main__":
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    short_options = "ho:sn:"
    long_options = ["help", "output=", "skipExisting", "numberOfQueries"]
    try:
        args, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    main(args)
