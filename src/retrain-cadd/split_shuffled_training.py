import os
import pickle
import csv
import time
import logging
import datetime
import collections
from tqdm import tqdm


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def get_batch_indexes(index_list,
                      batch_size,
                      output,
                      num_batches=None):

    num_indexes = len(index_list)
    amount_batches = (num_indexes // batch_size) + 1
    num_loops = min(
        num_batches, amount_batches) if num_batches else amount_batches
    batch_indexes = []

    for i in range(num_loops):
        start = (i) * batch_size
        end = min(num_indexes, start + batch_size)
        extraction = {
            'output': output + str(i + 1) + ".csv",
            'index_list': set(index_list[start:end])
        }
        batch_indexes.append(extraction)

    return(batch_indexes)


def generate_many_batches_fast(filename, sep, extraction):
    i = 0

    with open(filename) as input_file:
        # Obtain header of our master input file
        header = "ix" + sep + next(input_file)

        # Initialize batch values and headers of output files
        batch_values = {}
        for k in range(len(extraction)):
            batch_values[k] = {'ixs': [], 'rows': []}
            with open(extraction[k].get('output'), 'w') as out_f:
                out_f.write(header)

        # Read each line of master file and assign it to a batch
        row_number = 0
        for row in tqdm(input_file):
            not_found, li = True, 0
            # Loop thru batches until finding the right one
            while not_found:
                if row_number in extraction[li].get('index_list'):
                    batch_values[li]['ixs'].append(row_number)
                    batch_values[li]['rows'].append(str(row_number) + sep + row)
                    i += 1
                    not_found = False
                li += 1
                if li == len(extraction):
                    break

            if i >= 1000000:
                print("\nFinished ", i, " lines.")
                # Persist when having classified 10000 lines to free memory
                for j in range(len(extraction)):
                    with open(extraction[j].get('output'), 'a') as out_f:
                        out_f.writelines(batch_values[j]['rows'])
                    print("Added",
                          len(batch_values[j]['rows']),
                          "lines for batch", j + 1)

                    # Now we reduce the size of search for extraction
                    print("List had", len(extraction[j].get('index_list')),
                          "elements.")
                    prev = set(extraction[j].get('index_list'))
                    curr = prev - set(batch_values[j]['ixs'])
                    extraction[j]['index_list'] = set(curr)
                    print("Now it has", len(extraction[j].get('index_list')),
                          "elements.")
                    batch_values[j] = {'ixs': [], 'rows': []}
                curr = prev = None
                i = 0

            row_number += 1

        # Handle last batch
        print("\nFinished ", i, " lines.")
        for j in range(len(extraction)):
            with open(extraction[j].get('output'), 'a') as out_f:
                out_f.writelines(batch_values[j]['rows'])
            print("Added",
                  len(batch_values[j]['rows']),
                  "lines for batch", j + 1)


if __name__ == '__main__':
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler()
    handler = logging.FileHandler(
        "/s/project/kipoi-cadd/logs/splitting" +
        datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S") + ".log")
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info("Started script.")

    training_imputed = ("/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
                        "training_data.imputed.csv")
    output = ("/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
              "shuffle_splits/training/")
    shuffled_index_file = (
        "/s/project/kipoi-cadd/data/raw/v1.3/training_data/shuffle_splits/" +
        "shuffled_index.pickle")
    batches_index = (
        "/s/project/kipoi-cadd/data/raw/v1.3/training_data/shuffle_splits/" +
        "batches_index.pickle")
    training = ("/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
                "training_data.tsv")

    if not os.path.isfile(batches_index):
        with open(shuffled_index_file, 'rb') as f:
            shuffled_index = pickle.load(f)

        batches = get_batch_indexes(
            shuffled_index, 10000, output, num_batches=None)

        with open(batches_index, 'wb') as f:
            pickle.dump(batches, f)
    else:
        with open(batches_index, 'rb') as f:
            batches = pickle.load(f)

    logger.info("Finished loading/generating batches. Ended with, " +
                str(len(batches)) + ".")

    
    # test_batches = batches[:3]

    generate_many_batches_fast(training_imputed, ",", batches)

    end = time.time()
    logger.info(
        "Total elapsed time: {:.2f} minutes.".format((end - start) / 60))
