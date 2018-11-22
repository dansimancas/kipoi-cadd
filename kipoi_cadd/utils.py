import collections
import sys
import json


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def write_json(obj, fname, **kwargs):
    with open(fname, "w") as f:
        return json.dump(obj, f, **kwargs)
