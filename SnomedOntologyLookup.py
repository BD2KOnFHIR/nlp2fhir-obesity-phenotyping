#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SnomedOntologyLookup.py

Lookup the Snomed-CT codes found by JsonBasedReader.

Uses the Snomed-CT API to search for the parents and ancestors of Snomed-CT codes using the original Snomed Ontology.
Please refer to http://snomed.info for more information on terms of use
"""

import json
import requests
import time
from MLDataProcessing import save_to_json, load_dict_json, log_settings
import logging
from pathlib import Path

# base url for snomed api server http://_____________/api/v2/snomed/en-edition
base_uri = ''



def main(working_dir, depth=10):
    if depth > 0 and len(base_uri) == 0:
        exit("Please set base url for Snomed CT server in:" + __file__)

    log_settings(filename="SNOMED_LOOKUP.log", level=logging.INFO, filemode='w', stdout=True)
    working_dir = Path(str(working_dir))
    logging.info("Starting Snomed Lookup")

    # save progress after N base snomed ct lookups in case anything occurs during run
    SAVE_PROGRESS_EVERY_N = 10  # -1 disable save

    # dictionary with Snomed CT code to original text description of code
    # this will contain the snomed entries to search for
    snomed_file_to_lookup = working_dir /'data'/ 'snomed_found.json'
    logging.info("Loading data from: " + str(snomed_file_to_lookup))

    save_snomed_to_parents_file = working_dir / 'data' / 'snomed_parents_inferred.json'
    snomed_ancestor_file = working_dir / 'data' / ('snomed_ancestor_inferred.json')
    snomed_code_descriptions_from_query = working_dir / 'data' / 'snomed_description_from_query.json'

    snomed_to_description = load_dict_json(snomed_file_to_lookup)
    sctid_to_parents = load_dict_json(save_snomed_to_parents_file, create_local_if_not_found=True)
    snomed_to_ancestors = load_dict_json(snomed_ancestor_file, create_local_if_not_found=True)
    sctid_to_desc = load_dict_json(snomed_code_descriptions_from_query, create_local_if_not_found=True)

    list_of_snomed_to_lookup = list(snomed_to_description.keys())


    count = 0
    for sctid in list_of_snomed_to_lookup:
        if sctid not in sctid_to_desc:
            name = query_snomed_name(sctid)
            print(sctid, name)
            sctid_to_desc[sctid] = name
        count +=1
        if SAVE_PROGRESS_EVERY_N > 0 and count % int(SAVE_PROGRESS_EVERY_N) == 0 or count == len(list_of_snomed_to_lookup):  # save results once in a while
            save_to_json(sctid_to_desc, snomed_code_descriptions_from_query, indent=4)
            print("Saving results: ", count, "/", len(list_of_snomed_to_lookup), " entries processed.")



    count = 0
    for sctid in list_of_snomed_to_lookup:
        list_of_ancestors = list(set(get_snomed_ancestors(sctid, sctid_to_parents, sctid_to_desc, depth=depth, query_depth=depth)))
        if list_of_ancestors:
            snomed_to_ancestors[sctid] = list_of_ancestors
        count += 1
        if SAVE_PROGRESS_EVERY_N > 0 and count % int(SAVE_PROGRESS_EVERY_N) == 0:  # save results once in a while
            save_to_json(sctid_to_parents, save_snomed_to_parents_file, indent=4)
            save_to_json(snomed_to_ancestors, snomed_ancestor_file, indent=4)
            save_to_json(sctid_to_desc, snomed_code_descriptions_from_query, indent=4)
            print("Saving results: ", count, "/", len(list_of_snomed_to_lookup), " entries processed.")

    # save results

    save_to_json(sctid_to_parents, save_snomed_to_parents_file,indent=4,print_save_loc=True)
    save_to_json(snomed_to_ancestors, snomed_ancestor_file,indent=4,print_save_loc=True)
    save_to_json(sctid_to_desc, snomed_code_descriptions_from_query,indent=4,print_save_loc=True)







def get_snomed_ancestors(sctid, cached_parents, found_descriptions, depth=2, query_depth = 2):
    results = []
    if depth <= 0:
        return results

    if query_depth > 0:
        results = get_snomed_parents(sctid, cached_parents, found_descriptions, query=True)
    if query_depth <= 0:
        results = get_snomed_parents(sctid, cached_parents, found_descriptions, query=False)

    # if there are results try calling function again recursively to check for more parents/ancestors
    if results:
        for id in results:
            results = results + get_snomed_ancestors(id, cached_parents, found_descriptions, depth - 1, query_depth - 1)

    return results


def get_snomed_parents(sctid, cached_parents, found_descriptions, query = True):

    if sctid in cached_parents:
        return cached_parents[sctid]

    parents_list = []

    if query:
        parents = list(query_snomed_parents(sctid))
        if parents:
            for x in parents:
                parents_list.append(x[0])
                found_descriptions[x[0]] = x[1]
        cached_parents[sctid] = parents_list
        summary_text = "Snomed Query Results for: " + str(sctid) + str(parents_list)
        logging.info(summary_text)
    return parents_list


def query_snomed_parents(sctid):
    # Info on API:
    #     https://snomedctsnapshotapi.docs.apiary.io/#reference/releases/release-information
    '''
    :param sctid: Snomed-CT ID
    :return: inferred Parents of Snomed-CT ID
    '''
    time.sleep(0.2)    #limit rate of requests
    version = 'v20180131'
    form = 'inferred'   # stated or inferred
    global base_uri
    url = '{base_uri}/{release}/concepts/{sctid}/parents?form={form}'.format(base_uri = base_uri, release = version, sctid = str(sctid), form=form)
    response = requests.get(url)
    response.raise_for_status()
    parsed_json = json.loads(response.text)
    for i in range(len(parsed_json)):
        term = ''
        if 'preferredTerm' in parsed_json[i]:
            term = str(parsed_json[i]['preferredTerm'])
        if 'conceptId' in parsed_json[i]:
            sctid_found = str(parsed_json[i]['conceptId'])
            yield tuple([sctid_found, term])


def query_snomed_name(sctid):
    # Info on API:
    #     https://snomedctsnapshotapi.docs.apiary.io/#reference/releases/release-information
    '''
    :param sctid: Snomed-CT ID
    :return: inferred Parents of Snomed-CT ID
    '''
    time.sleep(1)    #limit rate of requests
    version = 'v20180131'
    form = 'inferred'   # stated or inferred
    global base_uri
    url = '{base_uri}/{release}/concepts/{sctid}'.format(base_uri = base_uri, release = version, sctid = str(sctid), form=form)
    response = requests.get(url)
    response.raise_for_status()
    try:
        parsed_json = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        print("error loading json at:", sctid, url)
        return ''

    if 'preferredTerm' in parsed_json.keys():
        return parsed_json['preferredTerm']
    else:
        return ''

if __name__ == '__main__':
    import sys

    try:
        main(sys.argv[1], int(sys.argv[2]))
    except IndexError:
        main(input("Please enter working directory: "), int(input("Please enter depth to search:")))
