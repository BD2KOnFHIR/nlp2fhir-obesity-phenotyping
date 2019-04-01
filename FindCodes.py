#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FindCodes.py

Finds the Snomed and RxNorm codes in json format FHIR Resource bundle.  These should be automatically generated
when running JsonBasedReader, so there should not be a need to run this.

Requires manually changing the input/output directories.
"""


from MLDataProcessing import save_to_json, log_settings


def main(resource_bundle_dir):
    from pathlib import Path
    import operator
    from collections import defaultdict

    snomed_savefile = 'snomed_found.json'
    rxnorm_savefile = 'rxcui_found.json'

    rxnorm_count = defaultdict(int)
    rxnorm_text = {}

    snomed_count = defaultdict(int)
    snomed_text = {}

    print("Loading data from: " + resource_bundle_dir)
    pathlist = Path(resource_bundle_dir).glob('*.json')
    for path in pathlist:

        with open(str(path), 'r', encoding='UTF-8') as fp:
            line = fp.readline()
            cnt = 1
            while line:
                line = line.strip()
                if line == '"system": "http://www.nlm.nih.gov/research/umls/rxnorm",':
                    line = fp.readline().strip()
                    if line[0:9] == '"code": "':
                        code = line[9:-1]
                        rxnorm_count[code] += 1
                        #try to get text
                        for _ in range(3):
                            line = fp.readline().strip()
                            if line[0:6] == '"text"':
                                text = line[9:-1]
                                rxnorm_text[code] = text
                                break
                if line == '"system": "http://snomed.info/sct",':
                    line = fp.readline().strip()
                    if line[0:9] == '"code": "':
                        code = line[9:-1]
                        snomed_count[code] += 1
                        #try to get text
                        for _ in range(3):
                            line = fp.readline().strip()
                            if line[0:6] == '"text"':
                                text = line[9:-1]
                                snomed_text[code] = text
                                break

                line = fp.readline()



    rxnorm_count = sorted(rxnorm_count.items(), key=operator.itemgetter(1), reverse=True)

    rxcui_description = {}
    for key, _ in rxnorm_count:
        rxcui_description[key]=rxnorm_text[key]


    snomed_count = sorted(snomed_count.items(), key=operator.itemgetter(1), reverse=True)

    snomed_description = {}
    for key, _ in snomed_count:
        snomed_description[key] = snomed_text[key]


    print("\nrx_norm_list:")
    print("rx Entries: ", len(rxnorm_count))
    print(rxnorm_count)
    print(rxcui_description)
    print("\nsnomed_list:")
    print("Snomed Entries: ", len(snomed_count))
    print(snomed_count)
    print(snomed_description)

    # save data
    save_to_json(snomed_description, snomed_savefile, indent=4)
    save_to_json(rxcui_description, rxnorm_savefile, indent=4)



    file_ascii = 'snomed_found.txt'
    write_plain = open(file_ascii, 'w')

    for key, value in snomed_description.items():
        str1 = str(key) + ': ' + str(value) + '\n'
        write_plain.write(str1)
    write_plain.close()


if __name__ == '__main__':
    log_settings(filename ="FindCodes.log")
    main(input("Enter Resource Bundle Dir: "))