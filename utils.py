import pickle


class Utils:
    @staticmethod
    def dump_to_pickle(obj_to_dump, pickle_name):
        pickle.dump(obj_to_dump, open(pickle_name, 'wb'))

    @staticmethod
    def load_from_pickle(pickle_name):
        return pickle.load(open(pickle_name, 'rb'))
