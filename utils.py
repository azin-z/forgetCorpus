import pickle
import functools
import time


class Utils:
    @staticmethod
    def dump_to_pickle(obj_to_dump, pickle_name):
        pickle.dump(obj_to_dump, open(pickle_name, 'wb'))

    @staticmethod
    def load_from_pickle(pickle_name):
        return pickle.load(open(pickle_name, 'rb'))


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
        print('starting', func.__name__)
        output = func(*args, **kwargs)
        print(bcolors.WARNING + "{} took {} seconds".format(func.__doc__, round(time.time() - start_time)) + bcolors.ENDC)
        return output
    return wrapper


def get_white_listed_ids():  # not pc
    white_listed_ids = set()
    forget_queries = Utils.load_from_pickle('forget-handwritten-dict.p')
    azzopardi_keys = Utils.load_from_pickle('azzopardi/pardi-model-3-with-coll-result-dict.p').keys()
    azzopardi_webis_ids = [get_webis_id(cw_id) for cw_id in azzopardi_keys]
    for key, value in forget_queries.items():
        if key in azzopardi_webis_ids:
            white_listed_ids.add(key)
    return white_listed_ids


def is_clueweb_id(name):
    return not name[:1].isnumeric()


def get_webis_id(clueweb_id):
    return Utils.load_from_pickle('clueweb_webis_map.p')[clueweb_id]


def calculate_mrr(result_pickle_name, restricted_list=None):
    restrict = restricted_list is not None
    result = Utils.load_from_pickle(result_pickle_name)
    mrr = 0
    hits = 0
    count = 0
    for key, value in list(result.items()):
        if is_clueweb_id(key):
            key = get_webis_id(key)
        if not restrict or (restrict and key in restricted_list):
            count += 1
            if value is not None:
                hits += 1
                mrr += 1/(value+1)
    try:
        print('mrr for all', mrr/count, 'for ', result_pickle_name)
        print('total hits', hits, 'out of', count)
        print('mrr for hits', mrr/hits, 'for ', result_pickle_name)
        return result_pickle_name, str(mrr/count), str(hits), str(mrr/hits)

    except Exception as e:
        print(e, 'occurred')
