import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
        Because numpy.ndarray is not a type that json knows how to handle, this approach "jsonify" numpy.ndarrays to
        list that can be saved to .json.

        Example:
            a = np.array([123])
            print(json.dumps({'aa':[2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder

        Copyright kalB (Dec 2017) from stackoverflow
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find(search_list, elem):
    """
        Search multiple elements in a list.

        :param search_list: a nested list.
        :param elem: a list.
        :return: a nested list, each with 0 or 1 for a searched element.
    """
    return [[i for i, x in enumerate(search_list) if e in x] for e in elem]