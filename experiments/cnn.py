import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Experiment
from utils import Utils, timing_decorator, calculate_mrr, get_white_listed_ids, get_webis_id
import torch.optim as optim
import anserini as anserini

import POS as pos
import NER as ner
max_len_item = 300
lr = 0.0001
kernel1 = 3
kernel2 = 5

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x10 square convolution
        # input is a nx10 matrix 3x10 convolution, 10 output channels
        # kernel
        self.conv1 = nn.Conv2d(1, 6, (10, kernel1))
        self.conv2 = nn.Conv2d(6, 16, (1, kernel2))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 72, 300*3)
        self.fc2 = nn.Linear(300*3, 300)
        # self.fc3 = nn.Linear(300, 300 * 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        a = self.conv1(x)
        # print("after first conv shape is {}".format(a.shape))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print("After max pool shape is {}".format(x.shape))
        x = x.reshape(-1, 16 * 72)
        # x = x.view(-1, self.num_flat_features(x))
        # print("After .view shape is {}".format(x.shape))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        # print("After .fc2 shape is {}".format(x.shape))
        # x = self.fc3(x)
        # x = x.reshape
        # print("After .fc3 shape is {}".format(x.shape))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class CNNExperminet(Experiment):
    def __init__(self, webiscorpus, train_samples=1000, use_ner=True, useTFIDF=True, use_noun=True, use_verb=True, use_adj=True, useHandwrittenAsGold=False):
        self.train_samples = train_samples
        self.model_name = 'cnn-model-'+ 'lr' + str(lr) + '-kernel1-' + str(kernel1) + '-kernel2-' + str(kernel2)
        self.train_samples = train_samples
        self.batch_size = 4
        self.max_len_item = max_len_item
        self.use_ner = use_ner
        self.use_verb = use_verb
        self.use_noun = use_noun
        self.use_adj = use_adj
        self.useTFIDF = useTFIDF
        self.silver_dict = Utils.load_from_pickle(
            'queries-silver.p')
        self.training_item_generator_funcssh  = webiscorpus.corpus_gen_non_white_listed
        # if useHandwrittenAsGold:
        #     self.training_item_generator_func = webiscorpus.corpus_gen_white_listed
        #     self.silver_dict = Utils.load_from_pickle('queries-handwritten.p')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(device)
        self.num_of_features = 10
        super(CNNExperminet, self).__init__(self.model_name, webiscorpus)

    def get_features_for_word(self, word):
        return [0] * self.num_of_features  #fix this

    def iterate_item_words(self, item):
        for word in anserini.tokenizeString(item['Content'] + item['Subject'], 'lucene'):
            yield word

    def get_all_item_terms(self, item):
        return anserini.tokenizeString(item['Content'] + item['Subject'], 'lucene')

    def process_word(self, i, term, block_type, pos_tag, entity_words, term_doc_count_dict, total_length):
        is_in_subject = int(block_type == 'subject')
        is_in_content = int(block_type == 'content')
        if self.use_noun:
            is_noun = int(pos_tag == 'NOUN')
        else:
            is_noun = 0
        if self.use_verb:
            is_verb = int(pos_tag == 'VERB')
        else:
            is_verb = 0
        if self.use_adj:
            is_adj = int(pos_tag == 'ADJ')
        else:
            is_adj = 0
        if self.use_ner:
            if term in entity_words:
                is_entity = entity_words[term]
            else:
                is_entity = 0
        else:
            is_entity = 0
        tf = float(anserini.get_term_coll_freq(term)) / 689710000
        try:
            idf = 1 / anserini.get_term_doc_freq(term)
        except:
            idf = 0
        tf_in_q = term_doc_count_dict[term] / total_length
        rel_pos = float(i) / total_length
        return [tf, idf, tf_in_q, rel_pos, tf*idf, is_in_content, is_noun, is_verb, is_adj, is_entity]

    def process_block(self, text, terms, block_type, term_doc_count_dict, total_length):
        pos_tags = pos.get_pos_tags(terms)
        entity_words = set()
        if self.use_ner:
            entity_words = ner.get_entities(text)
        item_features = []
        for i, (term, pos_tag) in enumerate(zip(terms, pos_tags)):
            item_features += [self.process_word(i, term, block_type, pos_tag, entity_words, term_doc_count_dict, total_length)]
        return item_features

    def process_query(self, item, terms):
        subject = item['Subject']
        content = item['Content']
        doc_vector = {}
        for i in terms:
            if i in doc_vector:
                doc_vector[i] += 1
            else:
                doc_vector[i] = 1
        subject_terms = anserini.tokenizeString(subject, 'lucene')
        content_terms = anserini.tokenizeString(content, 'lucene')
        total_length = len(subject_terms) + len(content_terms)
        if total_length > self.max_len_item:
            print('total length', total_length)
        item_features = self.process_block(subject, terms, 'subject', doc_vector, total_length)
        # item_features += self.process_block(content, content_terms, 'content', doc_vector, total_length)
        return item_features[:max_len_item]

    def get_features_for_terms(self, item, terms):
        feature_vector = [0] * self.max_len_item
        features = self.process_query(item, terms)
        # print(len(features), 'this is features true length')
        count = len(features)
        for i, feature in enumerate(features):
            feature_vector[i] = feature

        while count < self.max_len_item:
            feature_vector[count] = [0] * self.num_of_features
            count += 1
        return feature_vector

    def get_class_labels_for_item(self, item, terms):
        result = [0] * max_len_item
        for i, word in enumerate(terms):
            if i >= self.max_len_item:
                break
            result[i] = int(word in anserini.tokenizeString(self.silver_dict[item['Id']], 'lucene'))
        # print(len(terms), "this is true len of class labels")
        return result[:self.max_len_item]

    def get_all_data_in_batches(self):
        all_data = []
        for item in self.webiscorpus.corpus_gen_non_white_listed():
            terms = self.get_all_item_terms(item)
            term_features = self.get_features_for_terms(item, terms)
            class_labels = self.get_class_labels_for_item(item, terms)
            all_data.append((term_features, class_labels))

        num_train = int(0.8 * len(all_data))
        training_data = all_data[:num_train]
        val_data = all_data[num_train:]

        return self.get_batch_of_data(training_data), self.get_batch_of_data(val_data)

    def get_word_at_index_of_item(self, item, index):
        try:
            a = self.get_all_item_terms(item)
            return a[index]
        except Exception as e:
            print('exception', e, 'index is: ', index, 'length tokenized is', len(a))
            return ''

    def iterate_test_data(self):
        for item in self.webiscorpus.corpus_gen_white_listed():
            terms = self.get_all_item_terms(item)
            term_features = self.get_features_for_terms(item, terms)
            class_labels = self.get_class_labels_for_item(item, terms)
            yield term_features, class_labels, item

    def get_batch_of_data(self, data):
        batches_of_data = []
        batch_of_features = [0] * self.batch_size
        batch_of_labels = [0] * self.batch_size
        batch_index = 0
        for i, (term_features, class_labels) in enumerate(data):
            batch_of_features[batch_index] = term_features
            batch_of_labels[batch_index] = class_labels
            batch_index += 1
            if batch_index == self.batch_size:
                batch_index = 0
                batches_of_data.append((copy.copy(batch_of_features), copy.copy(batch_of_labels)))
        return batches_of_data

    def build_queries(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)
        total_correct = 0
        one = torch.ones(300).to(device)
        zero = torch.zeros(300).to(device)
        total = 0
        cnn_query_dict = {}
        model.load_state_dict(torch.load("BEST_CNN_MODEL"))
        with torch.no_grad():
            for data, gold, item in self.iterate_test_data():
                data = torch.tensor(data, dtype=torch.float, requires_grad=False).reshape(1, 1, 10, -1).to(device)
                gold = torch.tensor(gold, dtype=torch.float, requires_grad=False).to(device)
                output = model(data)
                # output = torch.where(output > 0.5, one, zero).to(device)
                output = output.reshape(-1)
                # idx = (output == 1).nonzero().flatten()
                # total += output.numel()
                # total_correct += torch.eq(gold, output).sum()
                query_words = set()
                output_sorted = output.tolist()
                output_sorted.sort()
                for index, num in enumerate(output_sorted):
                    if len(query_words) > 15:
                        break
                    query_words.add(self.get_word_at_index_of_item(item, index))
                query = " ".join(list(query_words))
                for word in self.silver_dict[item['Id']].split(" "):
                    if word not in query_words:
                        print(word)
                print('cnn***', query)
                print('gold***', self.silver_dict[item['Id']])
                print("")
                cnn_query_dict[item['Id']] = query
            # print(cnn_query_dict)
            self.dump_query_file(cnn_query_dict)
            # print("test accuracy is {}%".format(float(total_correct)*100/total))

        self.search_queries()
        _, mrr, _, _ = calculate_mrr(self.result_pickle_name, self.white_list)
        self.mrr = float(mrr)



    def run(self):

        # self.build_queries()
        # return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        one = torch.ones(4, 300).to(device)
        zero = torch.zeros(4, 300).to(device)

        self.net = self.net.to(device)
        self.net.train()

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # try:
        #     train_data = Utils.load_from_pickle('cnn-batch-train_data.p')
        #     valid_data = Utils.load_from_pickle('cnn-batch-val_data.p')
        # except:
        train_data, valid_data = self.get_all_data_in_batches()
        Utils.dump_to_pickle(train_data, 'cnn-batch-train_data.p')
        Utils.dump_to_pickle(valid_data, 'cnn-batch-val_data.p')
        # assert train_data_p == train_data
        # assert valid_data == valid_data_p

        best_dev_accuracy = 0
        for epoch in range(100):
            epoch_loss = 0
            total_train = 0
            correct_train = 0
            sample_count = 0

            for j, item in enumerate(train_data):
                batch, target = item
                sample_count += 1
                optimizer.zero_grad()

                batch = torch.tensor(batch, dtype=torch.float, requires_grad=False).reshape(4, 1, 10, -1).to(device)
                target = torch.tensor(target, dtype=torch.float, requires_grad=False).to(device)

                output = self.net(batch)
                loss = criterion(output, target)

                output = torch.where(output > 0.5, one, zero).to(device)
                # print("{} out of {} correct".format(torch.eq(target, output).sum(), 300*4))
                total_train += target[target == 1].numel()
                correct_train += torch.where(target == 1, output, zero).sum()

                loss.backward()
                # print("Loss for batch {} is {}".format(j, loss.item()))
                optimizer.step()  # Does the update
                epoch_loss += loss.item()

            with torch.no_grad():
                correct_dev = 0
                total_dev = 0
                for j, (dev_batch, dev_target) in enumerate(valid_data):
                    dev_batch = torch.tensor(dev_batch, dtype=torch.float, requires_grad=False).reshape(4, 1, 10, -1).to(device)
                    dev_target = torch.tensor(dev_target, dtype=torch.float, requires_grad=False).to(device)
                    dev_output = self.net(dev_batch)
                    dev_output = torch.where(dev_output > 0.5, one, zero).to(device)
                    total_dev += dev_target[dev_target == 1].numel()
                    correct_dev += torch.where(dev_target == 1, dev_output, zero).sum()
                dev_accuracy = float(correct_dev)/total_dev
                if epoch > 5:
                    best_dev_accuracy = max(dev_accuracy, best_dev_accuracy)
                    if dev_accuracy == best_dev_accuracy:
                        torch.save(self.net.state_dict(), "BEST_CNN_MODEL")
                        print("best dev accuracy at epoch {}".format(epoch))

            print('loss for epoch {} is {}, dev accuracy is {}, correct training number is {}'.format(epoch, epoch_loss/sample_count, dev_accuracy, correct_train))
            print('total training', total_train)
        self.build_queries()



