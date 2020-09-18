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


def calculate_mrr(result_pickle):
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
