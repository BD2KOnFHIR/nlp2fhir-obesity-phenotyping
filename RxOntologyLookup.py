#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RxOntologyLookup.py

Lookup the Rxcui found by JsonBasedReader.

Uses the RxNav, RxClass API to search for the ingredients of a Rxcui.  If the Rxcui can not be found (expired?), searches
will be run on the original extract text corresponding to the Rxcui to try to find the ingredients.
ATC levels 1-4 are searched for using the public API.  Level 5 (==drug) was not found and not included.

Disclaimer:  This product uses publicly available data from the U.S. National Library of Medicine (NLM),
    National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product
    and does not endorse or recommend this or any other product
"""



from re import split
import requests
import xml.etree.ElementTree
import time
from MLDataProcessing import save_to_json, load_dict_json, load_dict_pickle, pickle_something
from collections import defaultdict
import operator
from pathlib import Path

ingredients_name = {}
cache_cui_to_ingredients = {}
cache_cui_to_atc = {}
manual_ingredient_entries = {}
rxnorm_blank_search_results = []

def main(working_dir, find_ingreds = True, find_ATC = True, output_ATC_count=False):
    print("Starting RxNorm code lookup")
    working_dir = Path(working_dir)
    print("Loading Data from: " + str(working_dir))

    global cache_cui_to_ingredients, cache_cui_to_atc, manual_ingredient_entries
    rxnorm_savefile = working_dir / 'data' / 'rxcui_found.json'
    save_atc_file = working_dir / 'data' / 'rxcui_atc.json'
    ingredient_dict_file = working_dir / 'data' / 'rxcui_ingredient.json'
    manual_ingredient_entries_file = working_dir / 'data' / 'rxcui_ingred_manual_entries.json'
    ingredients_name_file = working_dir / 'data' / "rxcui_ingredient_names.json"
    rxcui_name_file = working_dir / 'data' / "rxcui_names.json"

    cache_cui_to_ingredients = load_dict_json(ingredient_dict_file, create_local_if_not_found=True)
    cache_cui_to_atc = load_dict_json(save_atc_file, create_local_if_not_found=True)
    manual_ingredient_entries = load_dict_json(manual_ingredient_entries_file, create_local_if_not_found=True)
    if not manual_ingredient_entries:
        print("Manual entries for expired etc. Rxcui can be added at:\n" + str(manual_ingredient_entries_file))
    ingredients_name = load_dict_json(ingredients_name_file, create_local_if_not_found=True)

    #rxcui_name = load_dict_json(rxcui_name_file, create_local_if_not_found=True)
    rxcui_name= {}

    # load rxcui to search for
    rxcui_to_lookup = load_dict_json(rxnorm_savefile, create_local_if_not_found=False)

    rxcui_to_atc = {}
    rxcui_to_ingredients = {}

    # count = 0
    # for rxcui in rxcui_to_lookup.keys():
    #     count += 1
    #     rxcui_name[rxcui] = query_rxnorm_name(rxcui)
    #     if count % 100 == 0 or count == len(rxcui_to_lookup):
    #         print("Working:", count)
    #         save_to_json(rxcui_name, rxcui_name_file, indent=4)
    #

    if find_ingreds:
        count = 0
        for rxcui in rxcui_to_lookup.keys():
            rxcui_to_ingredients[rxcui] = get_rxnorm_ingredients(rxcui)
            count += 1
            if count % 100 == 0 or count ==len(rxcui_to_lookup):
                save_to_json(rxcui_to_ingredients, ingredient_dict_file, indent=4)
                save_to_json(ingredients_name, ingredients_name_file, indent=4)
                print("Stage 1/4: Ingredients Lookup: ", count, "/", len(rxcui_to_lookup), " entries processed.")


        # try to look up missing entries because sometimes rxcui get retired
        for rxcui, ingredients in rxcui_to_ingredients.items():
            if ingredients:
                continue
            if rxcui in rxcui_to_lookup:
                original_text_from_FHIR = str(rxcui_to_lookup[rxcui])
            else:
                print("Can not find rxcui: ", rxcui)
                continue

            search_results = get_rxnorm_ingredients_using_multisearch(original_text_from_FHIR)
            search_results = list(set(search_results))
            if search_results:
                rxcui_to_ingredients[rxcui] = search_results
                print("Rxcui term found: ", rxcui, original_text_from_FHIR, search_results)
            else:
                print("Rxcui term could NOT be found:", original_text_from_FHIR)

    save_to_json(rxcui_to_ingredients, ingredient_dict_file, indent=4)
    save_to_json(ingredients_name, ingredients_name_file, indent=4)



    # try to find ingredients from original ingredients
    for rxcui, ingredients in rxcui_to_ingredients.items():
        new_ingredients = []
        for ingredient in ingredients:
            new_ingredients = new_ingredients + get_rxnorm_ingredients(ingredient)
        rxcui_to_ingredients[rxcui] = list(set(ingredients + new_ingredients))


    #find ATC codes
    if find_ATC:
        count = 0
        for code, ingredients in rxcui_to_ingredients.items():
            ATC = get_rxnorm_ATC(code)

            if not ATC:
                for ingredient in ingredients:
                    ATC = ATC + get_rxnorm_ATC(ingredient)

            if ATC:
                rxcui_to_atc[code] = list(set(ATC))
            else:
                print("ATC could not be found for ", code, '(%d/%d)' % (count, len(rxcui_to_ingredients)))
            count += 1
            if count % 100 == 0 or count ==len(rxcui_to_ingredients):
                save_to_json(rxcui_to_atc, save_atc_file, indent=4)
                print("Stage 4/4: ATC: ", count, "/", len(rxcui_to_ingredients), " entries processed.")


    #save data
    if find_ingreds:
        save_to_json(rxcui_to_ingredients, ingredient_dict_file,indent=4, print_save_loc=True)
        save_to_json(ingredients_name, ingredients_name_file, indent=4, print_save_loc=True)
    if find_ATC:
        save_to_json(rxcui_to_atc, save_atc_file, indent=4, print_save_loc=True)

    count = 0
    new_count = 0
    keys_to_del = []

    if output_ATC_count:
        ATC_struct = defaultdict(int)
        for key, ATCs in rxcui_to_atc.items():
            if not ATCs:
                continue
            for ATC in ATCs:
                for subcode in [ATC[0:1], ATC[0:3], ATC[0:4], ATC[0:5]]:
                    ATC_struct[subcode] += 1

        ATC_struct = sorted(ATC_struct.items(), key=operator.itemgetter(1), reverse=True)
        print("ATC count:")
        print(ATC_struct)




def is_int(string_:str):
  try:
    int(string_)
    return True
  except ValueError:
    return False



#attempts various searches to find the rxnorm ingredients using different iterations on the original extracted text
def get_rxnorm_ingredients_using_multisearch(term_to_search:str):
    results = []
    # try simple search with whole string
    simple_result = get_rxnorm_ingredients_using_search(term_to_search)
    if simple_result:
        return simple_result

    pos = term_to_search.rfind(':')
    if pos > 0:
        new_search_term = term_to_search[pos + 1:]
        results = get_rxnorm_ingredients_using_search(new_search_term)
        if results:
            return results

    # try filtering words:
    sep = split('\W+', term_to_search)
    longer_words = [word for word in sep if len(word) >= 3 and not is_int(word)]
    combined = ' '.join(longer_words)
    if not combined: # no valid words of sufficient length => give up
        return []

    results = get_rxnorm_ingredients_using_search(combined)
    if results:
        return results

    #try individual words
    for word in longer_words:
        results = results + get_rxnorm_ingredients_using_search(word)
    return results




def get_rxnorm_ingredients_using_search(term_to_search:str):
    global rxnorm_blank_search_results
    if term_to_search in rxnorm_blank_search_results:
        return []
    results = []
    search_results = list(query_rxnorm_ingredients_using_search(term_to_search))
    set_of_done = set()

    if search_results:
        for one_result in search_results:
            code = one_result[0]
            if code in set_of_done:
                continue
            set_of_done.add(code)
            results = results + get_rxnorm_ingredients(str(code))

    if not results:
        rxnorm_blank_search_results.append(term_to_search)
    return results


def query_rxnorm_ingredients_using_search(term):
    time.sleep(0.05)
    base_uri = 'http://rxnav.nlm.nih.gov/REST'
    url = '{base_uri}/approximateTerm?term={term}&maxEntries=4'.format(base_uri=base_uri, term=term)
    response = requests.get(url)
    tree = xml.etree.ElementTree.fromstring(response.text)
    xml_ingredients = tree.findall("./approximateGroup/candidate")
    for xml_ingredient in xml_ingredients:
        yield tuple(xml_ingredient.findtext(tag) for tag in ['rxcui', 'score', 'rank'])


def query_rxnorm_name(rxcui):
    time.sleep(0.05) #Limit API requests to max of 20/s

    base_uri = 'http://rxnav.nlm.nih.gov/REST'
    url = '{base_uri}/rxcui/{rxcui}/'.format(base_uri = base_uri, rxcui = rxcui)
    response = requests.get(url)
    tree = xml.etree.ElementTree.fromstring(response.text)
    for name_ in tree.findall("./idGroup/name"):
        return name_.text
    return ''


def get_rxnorm_ingredients(rxcui):
    global cache_cui_to_ingredients, ingredients_name, manual_ingredient_entries
    if rxcui in cache_cui_to_ingredients:
        return cache_cui_to_ingredients[rxcui]
    if rxcui in manual_ingredient_entries:
        return manual_ingredient_entries[rxcui]

    ingredients = list(query_rxnorm_ingredients(rxcui))
    ingredients_list = []
    if ingredients:
        for x in ingredients:
            ingredients_list.append(x[0])
            ingredients_name[x[0]] = x[1]
    cache_cui_to_ingredients[rxcui] = ingredients_list
    return ingredients_list


def query_rxnorm_ingredients(rxcui):
    time.sleep(0.05) #Limit API requests to max of 20/s

    base_uri = 'http://rxnav.nlm.nih.gov/REST'
    url = '{base_uri}/rxcui/{rxcui}/related?tty=IN'.format(base_uri = base_uri, rxcui = rxcui)
    response = requests.get(url)
    tree = xml.etree.ElementTree.fromstring(response.text)
    xml_ingredients = tree.findall("./allRelatedGroup/conceptGroup[tty='IN']/conceptProperties")
    for xml_ingredient in xml_ingredients:
        assert xml_ingredient.findtext('tty') == 'IN'
        yield tuple(xml_ingredient.findtext(tag) for tag in ['rxcui', 'name', 'umlscui'])
    xml_ingredients = tree.findall("./relatedGroup/conceptGroup[tty='IN']/conceptProperties")
    for xml_ingredient in xml_ingredients:
        assert xml_ingredient.findtext('tty') == 'IN'
        yield tuple(xml_ingredient.findtext(tag) for tag in ['rxcui', 'name', 'umlscui'])


def get_rxnorm_ATC(rxcui):
    global cache_cui_to_atc
    if rxcui in cache_cui_to_atc:
        return cache_cui_to_atc[rxcui]

    ATCs = list(query_rxnorm_ATC(rxcui))
    atc_list = []
    if ATCs:
        for atc in ATCs:
            atc_list.append(atc[0])
    cache_cui_to_atc[rxcui] = atc_list
    return atc_list


def query_rxnorm_ATC(rxcui):
    time.sleep(0.05)    #limit requests under 20 requests/s

    base_uri = 'https://rxnav.nlm.nih.gov/REST'
    url = '{base_uri}/rxclass/class/byRxcui?rxcui={rxcui}&relaSource=ATC'.format(base_uri = base_uri, rxcui = rxcui)
    response = requests.get(url)
    tree = xml.etree.ElementTree.fromstring(response.text)
    xml_ATC = tree.findall("./rxclassDrugInfoList/rxclassDrugInfo/rxclassMinConceptItem[classType='ATC1-4']")
    for ATC in xml_ATC:
        assert ATC.findtext('classType') == 'ATC1-4'
        yield tuple(ATC.findtext(tag) for tag in ['classId', 'className', 'classType'])


if __name__ == '__main__':
    import sys
    if(len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main(input("Please enter working directory: "))