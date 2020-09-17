import json
import argparse
from datetime import datetime
import glob
import os
import time
import pickle
import random
from tqdm import tqdm
from collections import Counter

from utils import Utils, bcolors, timing_decorator
from Azzopardi import AzzopardiFunctions
import Classification as Classification
import anserini as anserini
from CommandLineActions import CommandLineActions
import NER as ner


class WebisCorpus:
    def __init__(self, args):
        self.skip_existing = args.skip
        self.num_of_queries = args.num
        self.data_list = None
        self.output_file_name = args.output
        self.showAnswer = args.showAnswer

        if args.index == 'mini':
            self.searcher = anserini.JSimpleSearcher(anserini.JString('/home/azahrayi/memory-augmentation/query-simulation/ForgetCorpus/webis-docs/index.pos+docvectors+raw+porter'))
            self.reader = anserini.JIndexReaderUtils.getReader(anserini.JString('/home/azahrayi/memory-augmentation/query-simulation/ForgetCorpus/webis-docs/index.pos+docvectors+raw'))
        else:
            self.searcher = anserini.JSimpleSearcher(
                anserini.JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
            self.reader = anserini.JIndexReaderUtils.getReader(
                anserini.JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
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

    def classify(self, train=True, makeQueries=False, train_samples=1000):
        classification = Classification()
        query_dict = {}
        silver_dict = Utils.load_from_pickle('id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')
        if train:
            for i, item in enumerate(self.corpus_gen()):
                if i > train_samples:
                    break
                classification.process_query(item['Subject'], item['Content'], silver_dict[item['Id']])
            classification.train()
            classification.save_model(train_samples)
        else:
            classification.load_model('classifiermodels/clf-model-no-common-words1000.p')
        for i, item in enumerate(self.corpus_gen()):
            # if i > 1000:
            #     break
            full_terms = classification.process_query(item['Subject'], item['Content'], silver_dict[item['Id']])
            if makeQueries:
                result = classification.predict().tolist()
                query_terms = []
                while len(query_terms) < 10 and len(result):
                    picked_word_index = result.index(max(result))
                    picked_word = full_terms[picked_word_index]
                    result.pop(picked_word_index)
                    full_terms.pop(picked_word_index)
                    if picked_word not in query_terms:
                        query_terms.append(picked_word)
                # print('result', result)
                # query_terms = [full_terms[i] for i in range(len(result)) if result[i] == 1]
                query_terms = ' '.join(set(query_terms))
                print('classifier:', query_terms)
                query_dict[item['Id']] = query_terms
            # print(full_terms)
                print('silver:', silver_dict[item['Id']])
            # # print(result)
        if makeQueries:
            Utils.dump_to_pickle(query_dict, 'id-svm-made-query-pos-ner-top-10-' + str(train_samples) +'.p')
        else:
            result = classification.predict()
            classification.getAccuracy(result)

    def build_document_dictionary(self):
        id_doc_dict = {}
        errorCount = 0
        import requests
        for item in tqdm(self.corpus_gen()):
            link = item['KnownItemUrl']
            try:
                response = requests.get(
                    'https://en.wikipedia.org/w/api.php',
                    params={
                        'action': 'query',
                        'format': 'json',
                        'titles': link.split('/')[-1],
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True,
                    }).json()
                id_doc_dict[item['Id']] = next(iter(response['query']['pages'].values()))['extract']
            except:
                errorCount += 1
        pickle.dump(id_doc_dict, open('id-doc-text-dict' + '.p', "wb"))
        # 262 error count
        print(errorCount)

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

    def get_term_coll_freq(self, term):
        jterm = anserini.JTerm("contents", term.lower())
        cf = self.reader.totalTermFreq(jterm)
        return cf

    def get_term_doc_freq(self, term):
        jterm = anserini.JTerm("contents", term)
        df = self.reader.docFreq(jterm)
        return df

    def make_query_with_overlapping_terms(self, pickle1, pickle2, resultname, analyzer=None):
        p1 = Utils.load_from_pickle(pickle1)
        p2 = Utils.load_from_pickle(pickle2)
        print(p2)
        gold_doc_dict = Utils.load_from_pickle('id-gold-doc-dict.p')
        terms_in_common_dict = {}
        for key, value in p1.items():
            counter = 0
            terms_in_common = []
            terms1 = list(set(anserini.tokenizeString(value, analyzer)))
            try:
                terms2 = list(set(anserini.tokenizeString(p2[gold_doc_dict[key]], analyzer)))
                # print(terms2)
            except:
                continue
            for term in terms1:
                if term in terms2:
                    terms_in_common.append(term)
                    counter += 1
            terms_in_common_dict[key] = ' '.join(terms_in_common)
        print(terms_in_common_dict)
        Utils.dump_to_pickle(terms_in_common_dict, 'id-terms-in-common-' + resultname + '-dict.p')

    def select_top_words(self, text, k, type):
        """selecting top words"""
        terms = list(set(anserini.tokenizeString(text, 'lucene')))
        if type == 'random':
            return ' '.join(random.sample(terms, k))
        scores = []
        for i, term in enumerate(terms):
            try:
                tf = self.get_term_coll_freq(term)
                idf = 1/self.get_term_doc_freq(term)
            except ZeroDivisionError:
                idf = 0
                if term in anserini.stopwords_temp:
                    terms.pop(i)
                    continue
            except Exception as e:
                print(e)
                continue
            if type == 'tf-idf':
                tf_idf = tf * idf
                scores.append(tf_idf)
            if type == 'tf':
                scores.append(tf)
            if type == 'idf':
                scores.append(idf)
        picked_words = []
        if type == 'tf':
            min_or_max_function = min
        else:
            min_or_max_function = max
        while len(picked_words) < k and len(scores):
            picked_word_index = scores.index(min_or_max_function(scores))
            picked_word = terms[picked_word_index]
            scores.pop(picked_word_index)
            terms.pop(picked_word_index)
            if picked_word not in picked_words and \
                            picked_word != 'remember' and picked_word != 'forget':
                picked_words.append(picked_word)
        return ' '.join(picked_words)

    def get_webis_docs(self):
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
                            print('HUUURRAY', docid)
                            webis_docid_document_content_dict[docid] = parsed.get('contents')
                    except:
                        print('null')
        pickle.dump(webis_docid_document_content_dict, open('cleuweb-webis-id-doc-content-dict' + '.p', "wb"))

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

    def azzopardi_models(self, option):
        if option == 0:
            id_doc_text = Utils.load_from_pickle('cleuweb-webis-id-doc-content-dict.p')
            azzopardifuncs = AzzopardiFunctions()
            for id in tqdm(id_doc_text.keys()):
                doc_vector = {}
                for i in anserini.tokenizeString(id_doc_text[id], 'lucene'):
                    if i in doc_vector:
                        doc_vector[i] += 1
                    else:
                        doc_vector[i] = 1
                print(self.get_doc_url(id))
                try:
                    azzopardifuncs.make_query(id, doc_vector, 10)
                except Exception as e:
                    print('error ', e, 'occured in processing', id)
        else:
            result = self.search_queries('azzopardi/e-id-pardi-model-' + str(option) + '-with-coll-queries.p', rm3=False, miniIndex=True)
            Utils.dump_to_pickle(result, 'azzopardi/pardi-model-' + str(option) + '-with-coll-result-dict.p' )
            self.calculate_mrr('azzopardi/pardi-model-' + str(option) + '-with-coll-result-dict.p')



    @timing_decorator
    def build_query_dicts(self, type, k=None):
        """building dictionaries for queries"""
        id_query_dict = {}
        for item in tqdm(self.corpus_gen()):
            id = item['Id']
            # gold_doc_id = item['KnownItemId']
            # id_gold_doc_dict[id] = gold_doc_id
            try:
                # if type == 'ne':

                if type == 'automatic':
                    id_query_dict[id] = item['Subject'] + item['Content']
                if type == 'handwritten':
                    query = item['ForgetQuery']
                    if query == 'NOTFORGETQUERY':
                        continue
                    id_query_dict[id] = query
                if type == 'url':
                    id_query_dict[id] = item['KnownItemUrl'].split('/')[-1].replace('_', ' ').replace('-', ' ').replace('.html', '').replace('.jsp', '').replace('.htm','').replace('.php', '').lower()
                else:
                    id_query_dict[id] = self.select_top_words(item['Subject'] + item['Content'], k, type)
            except:
                continue
        Utils.dump_to_pickle(id_query_dict, 'id-' + type + '-top-' + str(k) + '-dict.p')
        return id_query_dict

    @timing_decorator
    def grid_build_search(self, type_str, k):
        self.build_query_dicts(type_str, k)
        result = self.search_queries('id-' + type_str + '-top-' + str(k) + '-dict.p')
        Utils.dump_to_pickle(result, 'result-dict-' + type_str + '-top-' + str(k) + '.p')

    @timing_decorator
    def search_queries(self, query_pickle, rm3=False, miniIndex=False):
        """Searching clueweb by query file"""
        queries = pickle.load(open(query_pickle, 'rb'))
        result_dict = {}
        jqueries = anserini.JList()
        ids = anserini.JList()
        gold_doc_dict = pickle.load(open('id-gold-doc-dict.p', 'rb'))
        for i, query in enumerate(list(queries.values())):
            jqueries.add(query)
        for i, id in enumerate(list(queries.keys())):
            ids.add(id)
        if rm3:
            self.searcher.setRM3Reranker()
        # searcher.setLMDirichletSimilarity(2000)
        # print('rm3 status', searcher.setRM3Reranker())
        results = self.searcher.batchSearch(jqueries, ids, 500, 40)
        resultsKeySet = results.keySet().toArray()
        for resultKey in resultsKeySet:
            for i, result in enumerate(results.get(resultKey)):
                if miniIndex:
                    if result.docid == resultKey:
                        result_dict[resultKey] = i
                        break
                else:
                    if result.docid == gold_doc_dict[resultKey]:
                        result_dict[resultKey] = i
                        break
            if resultKey not in result_dict:
                result_dict[resultKey] = None
        return result_dict

    @staticmethod
    def calculate_mrr(result_pickle, query_pickle=None):
        result = Utils.load_from_pickle(result_pickle)
        mrr = 0
        hits = 0
        for key, value in list(result.items()):
            if value is not None:
                hits += 1
                mrr += 1/(value+1)
        try:
            print('mrr for hits', mrr/hits, 'for ', result_pickle)
            print('mrr for all', mrr/len(result), 'for ', result_pickle)
            print('total hits', hits, 'out of', len(result.items()))
        except Exception as e:
            print(e, 'occurred')

    def write_corpus_to_file(self):
        with open(self.output_file_name, 'w') as outputCorpusFile:
            outputCorpusFile.write(json.dumps(self.data_list))

    def print_queries(self, pickle1, pickle2, pickle3):
        p1 = Utils.load_from_pickle(pickle1)
        p2 = Utils.load_from_pickle(pickle2)
        p3 = Utils.load_from_pickle(pickle3)
        for key, value in p1.items():
            print(key)
            print(value)
            print('---')
            print(p2[key])
            print('---')
            print(p3[key])
            print('****')


@timing_decorator
def make_random_mini_collection_vocab(type):
    #total 689710000
    start_t = time.time()
    print('building mini collection')
    # collection_term_count_dict = Utils.load_from_pickle('clueweb-mini-collection_term_count_dict.p')
    # print(len(collection_term_count_dict))
    collection_term_count_dict = pickle.load(open('clueweb-collection_term_count_dict.p', "rb"))
    print('loaded full collection took', time.time() - start_t)
    mini_collection_term_count_dict = {}
    if type == 1:
        start_t = time.time()
        for i, (term, count) in tqdm(enumerate(collection_term_count_dict.items())):
            if i % 10000 == 0:
                mini_collection_term_count_dict[term] = count
        print('time took to make mini style 1 is', time.time() - start_t)
    Utils.dump_to_pickle(mini_collection_term_count_dict, 'clueweb-mini-collection_term_count_dict-style-1.p')
    mini_collection_term_count_dict_2 = {}

    if type == 2:
        start_t = time.time()
        for i, (term, count) in enumerate(list(collection_term_count_dict.items())[0:689710000:10000]):
            if i % 10000 == 0:
                mini_collection_term_count_dict_2[term] = count
        print('time took to make mini style 2 is', time.time() - start_t)
    Utils.dump_to_pickle(mini_collection_term_count_dict_2, 'clueweb-mini-collection_term_count_dict-style-2.p')
    return


def dump_collection_and_vocab_size(corpus_name):
    reader = JIndexReaderUtils.getReader(JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
    iterator = JIndexReaderUtils.getTerms(reader)
    start_time = time.time()
    counter = 0
    collection_size = 0
    coll_term_count_dict = Counter()
    with tqdm(total=689710000) as pbar:
        while iterator.hasNext():
            pbar.update(1)
            counter += 1
            index_term = iterator.next()
            collection_size += index_term.getTotalTF()
            coll_term_count_dict[index_term.getTerm()] = index_term.getTotalTF()
            if counter % 1000000 == 0:
                pickle.dump(collection_size, open('clueweb-collection_vocab_size.p', 'wb'))
                pickle.dump(counter, open('clueweb-collection-size.p', 'wb'))
                pickle.dump(coll_term_count_dict, open('clueweb-collection_term_count_dict.p', 'wb'))
    pickle.dump(collection_size, open('clueweb-collection_vocab_size.p', 'wb'))
    pickle.dump(counter, open('clueweb-collection-size.p', 'wb'))
    pickle.dump(coll_term_count_dict, open('clueweb--collection_term_count_dict.p', 'wb'))
    print("DONE")
    coll_term_prob_dict = {}
    for term, count in coll_term_count_dict.items():
        coll_term_prob_dict[term] = float(lambda_val * count) / float(collection_size)
    print(len(coll_term_prob_dict))
    pickle.dump(coll_term_prob_dict, open('clueweb-collection_term_prob_dict.p', 'wb'))


def remove_stopwords():
    query_common = Utils.load_from_pickle('id-terms-in-common-no-stopwords-automatic-doc-lucene-dict.p')
    new_dict = {}
    average_len = 0
    max_len = 0
    total_queries = len(query_common.items())
    additional_stopwords = ('i', 'what', 'who', 'which', 'so', 'you', 'would', 'me', 'when', 'your', 'can', 'my', 'about', 'from', 'all')
    for i, (key, value) in enumerate(query_common.items()):
        new_q = [term for term in value.split(' ') if term not in additional_stopwords]
        average_len += len(new_q)
        if max_len < len(new_q):
            max_len = len(new_q)
        new_dict[key] = ' '.join(new_q)
    print(total_queries)
    print('average length', average_len/total_queries)
    print('max length', max_len)
    Utils.dump_to_pickle(new_dict, 'id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')

def main(args):
    corpus = WebisCorpus(args)
    if args.experiment == 'classifier':
        train_samples = 2755
        corpus.classify(train=True, makeQueries=True, train_samples=train_samples)
        result = corpus.search_queries('id-svm-made-query-pos-ner-top-10-' + str(train_samples) + '.p')
        Utils.dump_to_pickle(result, 'svm-made-query-pos-ner-top-10-' + str(train_samples) + '-result.p')
        corpus.calculate_mrr('svm-made-query-pos-ner-top-10-' + str(train_samples) + '-result.p')
        # corpus.classify(makeQueries=True)
        return
    if args.experiment == 'term-cf':
        print(corpus.get_term_coll_freq('i'), 'i')
        print(corpus.get_term_coll_freq('johnny'), 'johnny')
        print(corpus.get_term_doc_freq('johnny'), 'johnny df')
        print(corpus.get_term_coll_freq(args.term), args.term, 'df')
        print(corpus.get_term_doc_freq(args.term), args.term, 'df')
        return
    if args.experiment == 'search-no-stopwords-no-common-words':
        result = corpus.search_queries('id-terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict.p')
        Utils.dump_to_pickle(result, 'terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict-result.p')
        corpus.calculate_mrr('terms-in-common-no-stopwords-and-common-words-automatic-doc-lucene-dict-result.p')
        return
    if args.experiment == 'search-ner-only':
        queries = {}
        for item in corpus.corpus_gen():
            queries[item['Id']] = ' '.join(list(ner.get_entities(item['Subject'] + item['Content'])))
        Utils.dump_to_pickle(queries, 'id-only-ne.p')
        print(queries)
        # queries = corpus.build_query_dicts('ne')
        result = corpus.search_queries('id-only-ne.p')
        Utils.dump_to_pickle(result, 'only-ne-result.p')
        corpus.calculate_mrr('only-ne-result.p')
        return
    if args.experiment == 'search-automatic':
        result = corpus.search_queries('id-automatic-dict.p')
        Utils.dump_to_pickle(result, 'automatic-result.p')
        corpus.calculate_mrr('automatic-result.p')
        return
    if args.experiment == 'mini-coll':
        make_random_mini_collection_vocab(1)
        return
    if args.experiment == 'categories':
        corpus.categories()
        return
    if args.experiment == "build-pardi-queries":
        corpus.azzopardi_models(0)
        return
    if args.experiment == "pardi":
        if args.model is None:
            print('state model number with -mod or --model: 1, 2, 3 or 4')
            return
        corpus.azzopardi_models(args.model)
        return
    if args.experiment == "build-mini-collection":
        make_random_mini_collection_vocab(2)
        return
    if args.experiment == "common-low-tf":
        result = corpus.search_queries('id-terms-in-common-complete-tf-dict.p')
        pickle.dump(result, open('terms-in-common-complete-tf-result-dict.p', "wb"))
        return
    if args.experiment == "search-forget":
        result = corpus.search_queries('id-forget-dict.p')
        Utils.dump_to_pickle(result, 'forget-result-dict.p')
        return
    if args.experiment == "search-forget-mini":
        result = corpus.search_queries('id-forget-dict.p')
        pickle.dump(result, open('forget-mini-index-result-dict.p', "wb"))
        return
    if args.experiment == "search-forget-mini-rm3":
        result = corpus.search_queries('id-forget-dict.p', rm3=True, mini=True)
        pickle.dump(result, open('forget-mini-index-rm3-result-dict.p', "wb"))
        return
    if args.experiment == "search-automatic-mini":
        result = corpus.search_queries('id-automatic-dict.p', mini=True)
        pickle.dump(result, open('automatic-mini-index-result-dict.p', "wb"))
        return
    if args.experiment == "search-automatic-rm3-mini":
        result = corpus.search_queries('id-automatic-dict.p', rm3=True)
        pickle.dump(result, open('automatic-mini-index-rm3-result-dict.p', "wb"))
        return
    if args.experiment == "search-silver-mini":
        result = corpus.search_queries('id-terms-in-common-automatoc-complete-dict.p', rm3=True, mini=True)
        pickle.dump(result, open('terms-in-common-mini-index-result-dict.p', "wb"))
        return
    if args.experiment == "overlap-terms-forget":
        corpus.make_query_with_overlapping_terms('id-forget-dict.p', 'id-doc-text-dict.p', 'forget-doc', args.analyzer)
        result = corpus.search_queries('id-terms-in-common-forget-doc-dict.p')
        Utils.dump_to_pickle(result, 'result-dict-terms-in-common-forget-doc.p')
        corpus.calculate_mrr('result-dict-terms-in-common-forget-doc.p')
        corpus.calculate_mrr('forget-top-500-result-dict.p')
        return
    if args.experiment == "overlap-terms-automatic":
        # corpus.build_query_dicts()
        # corpus.count_overlap_of_terms('id-automatic-dict.p', 'cleuweb-webis-id-doc-content-dict.p', 'automatic-doc-' + args.analyzer, args.analyzer)
        result = corpus.search_queries('id-terms-in-common-automatic-doc-'+args.analyzer+'-dict.p')
        Utils.dump_to_pickle(result, 'result-dict-terms-in-common-automatic-doc-' + args.analyzer + 'analyzer' + '.p')
        corpus.calculate_mrr('result-dict-terms-in-common-automatic-doc-' + args.analyzer + 'analyzer' + '.p') #so weird...
        # # corpus.calculate_mrr('terms-in-common-automatoc-complete-result-dict.p')
        corpus.calculate_mrr('automatic-top-500-result-dict.p')
        return
    if args.experiment == "tf-idf-queries":
        for k in range(5, 30, 5):
            corpus.grid_build_search('tf-idf', k)
        for k in range(5, 30, 5):
            corpus.calculate_mrr('result-dict-tf-idf-top-' + str(k) + '.p')
        return
    if args.experiment == 'random-grid':
        for k in range(5, 30, 5):
            corpus.grid_build_search('random', k)
        for k in range(5, 30, 5):
            corpus.calculate_mrr('result-dict-random-top-' + str(k) + '.p')
    if args.experiment == "tf-queries":
        for k in range(5, 30, 5):
            corpus.grid_build_search('tf', k)
        for k in range(5, 30, 5):
            corpus.calculate_mrr('result-dict-tf-top-' + str(k) + '.p')
        return
    if args.experiment == "idf-queries":
        corpus.grid_build_search('idf', 3821)
        corpus.calculate_mrr('result-dict-idf-top-' + str(3821) + '.p')
        return
    if args.search:
        print('search')
        if args.resultpickle:
            corpus.calculate_mrr(args.resultpickle)
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
    parser.add_argument('--mrr', '-mrr', type=bool, nargs='?', const=True, required=False,
                        default=False,
                        help='calculate mrr for given search pickle')
    parser.add_argument('--resultpickle', '-rpickle', type=str, required=False,
                        default=False,
                        help='result pickle file to calculate mrr for')
    parser.add_argument('--experiment', '-exp', type=str, required=False,
                        default=None,
                        help='specify experiment to run')
    parser.add_argument('--term', '-te', type=str, required=False,
                        default='i',
                        help='specify experiment to run')
    parser.add_argument('--model', '-mod', type=int, required=False,
                        default=1,
                        help='specify experiment to run')
    parser.add_argument('--analyzer', '-ana', type=str, required=False,
                        default='none',
                        help='specify tokenizer to use to, none or lucene ')
    parser.add_argument('--index', '-ind', type=str, required=False,
                        default='full',
                        help='specify tokenizer to use to, none or lucene ')
    main(parser.parse_args())
