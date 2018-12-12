"""Functions to efficiently load batches of data
"""
from tqdm import tqdm


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
