#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""JsonBasedReader.py

Parses through FHIR Resource Bundle in json format.
Finds Lion-C sections and uuid's associated with them.
Finds Snomed-CT and RxNorm Rxcui's and associates them with the sections based on uuid.
The negation clause is currently set to use 'abatementString' in the current FHIR2ML pipeline.
The addition/changes to negation and uncertainty will require updating this script.

Outputs csv style '.txt' file with Lion-C section + Snomed/Rxcui along with the count of appearances
and the number of negations associated with it.

Also saves the Snomed-CT and Rxcui codes found and shows a basic summary of results of Lion-C sections and their contents.

This is set to only detect codes in defined types of resource content.  The addition or use of other types of content
will require updating to determine the location of the main codes.
"""


import logging
import string
import os
import re
from collections import OrderedDict
from collections import defaultdict
from enum import Enum
from operator import itemgetter
from pathlib import Path
from statistics import mean

from MLDataProcessing import save_to_json, load_dict_json, log_settings

LOINC_SECT_CODES = "\d{2,6}-\d"
FHIR_RESOURCE_CODES = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

SNOMED_REFERENCE = 'http://snomed.info/sct'
RXNORM_REFERENCE = 'http://www.nlm.nih.gov/research/umls/rxnorm'
NEGATION_CLAUSE = 'abatementString'   # temporarily used key for negation (will need to update if FHIR changes)
#UNCERTAINTY_CLAUSE = ''    # not yet implemented

# regular expressions
re_loinc = re.compile(LOINC_SECT_CODES)
re_fhir_rsc = re.compile(FHIR_RESOURCE_CODES)

INCL_ADDTL_CODES = True # Include additional associated codes (relating to main code) in a resource.


def main(data_dir=None, work_dir=None):
    while data_dir is None or Path(data_dir).exists() is False:
        print("Unable to locate directory.")
        data_dir = input("Please enter data directory (FHIR JSON Resource Bundle): ")
    while work_dir is None or Path(work_dir).exists() is False:
        print("Unable to locate directory.")
        work_dir = input("Please enter working directory: ")

    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    pathlist = Path(data_dir).glob('*.json')

    log_settings(filename="json_based_reader.log", filemode='w')

    os.makedirs(work_dir / "output", exist_ok=True)
    print("Trying to load data from: " + str(data_dir))
    print("Working Directory: " + str(work_dir))

    lionc_words_record = defaultdict(list)
    lionc_characters_record = defaultdict(list)
    lionc_snomed_count_record = defaultdict(list)
    lionc_rxnorm_count_record = defaultdict(list)

    # track all the snomed ct, rxcui encountered
    sct_to_desc = {}
    rxcui_to_desc = {}

    for path in pathlist:
        path_in_str = str(path)
        report = load_dict_json(path_in_str)

        resource_to_section = {}
        section_to_resource = defaultdict(list)
        code_counts = defaultdict(int)
        code_negation_counts = defaultdict(int)

        lionc_words = defaultdict(int)
        lionc_characters = defaultdict(int)
        lionc_snomed_count = defaultdict(int)
        lionc_rxnorm_count = defaultdict(int)

        try:
            sections_and_references = report['entry'][0]['resource']['section']
        except KeyError:
            resource_to_section = defaultdict(lambda: '00000:0')

        # Read through first section defining Lion-C sections and references to uuid
        for lionc in sections_and_references:
            lionc_code = lionc['code']['coding'][0]['code']
            if not re_loinc.match(lionc_code):
                lionc_code = '00000-0'

            lionc_text = lionc['text']['div']
            word_char_count = text_word_counter(lionc_text)
            lionc_words[lionc_code] += word_char_count[0]
            lionc_characters[lionc_code] += word_char_count[1]

            if 'entry' in lionc:
                for item in lionc['entry']:  # references
                    reference = re_fhir_rsc.findall(item['reference'])[0]
                    resource_to_section[reference] = lionc_code
                    section_to_resource[lionc_code].append(reference)

        for i in range(1, len(report['entry'])):
            try:
                a_resource = report['entry'][i]['resource']
                resource_type = a_resource['resourceType']
                uuid = a_resource['id']
            except Exception as e:
                print(type(e))

            # Add new resource types if necessary.  Code locations need to be manually defined.
            try:
                if resource_type == 'Condition':
                    cct = ConditionEntry(a_resource)
                elif resource_type == 'FamilyMemberHistory':
                    cct = FamilyHistoryEntry(a_resource)
                elif resource_type == 'Medication':
                    cct = MedicationEntry(a_resource)
                elif resource_type == 'MedicationStatement':
                    cct = MedicationStatementEntry(a_resource)
                elif resource_type == 'Procedure':
                    cct = ProcedureEntry(a_resource)
                else:
                    print(resource_type, " was not included.")
            except KeyError as err:
                logging.info(err)
                logging.info("code value (rxcui/sct) not found in file:" + path_in_str)
                logging.info(str(a_resource))

            # print(cct.return_codes())
            section = find_section_for_uuid(cct.uuid, resource_to_section)
            snomed_rxn_counts = cct.code_type_counts()
            lionc_snomed_count[section] += snomed_rxn_counts[0]
            lionc_rxnorm_count[section] += snomed_rxn_counts[1]

            combined_section_with_code, negation_status = entry_to_codes(cct, resource_to_section, sct_to_desc=sct_to_desc,
                                                        rxcui_to_desc=rxcui_to_desc, incl_addtl_codes=INCL_ADDTL_CODES)
            for code in combined_section_with_code:
                code_counts[code] += 1
                if negation_status:
                    code_negation_counts[code] += 1


        for lionc in section_to_resource:
            # save to results for all records
            lionc_words_record[lionc].append(lionc_words[lionc])
            lionc_characters_record[lionc].append(lionc_characters[lionc])
            lionc_snomed_count_record[lionc].append(lionc_snomed_count[lionc])
            lionc_rxnorm_count_record[lionc].append(lionc_rxnorm_count[lionc])

        code_counts = OrderedDict(sorted(code_counts.items(), key=itemgetter(1), reverse=True))

        # output file (csv format with original file name)
        file_name = path.stem
        if file_name.find('.')> 0:
            file_name = file_name[:(file_name.find('.'))]
        output_path = work_dir / 'output' / (file_name + '.txt')
        with open(output_path, 'w') as output:
            output.write("code,count,negation\n")
            for k, v in code_counts.items():
                text = k + "," + str(v) + "," + str(code_negation_counts[k]) + "\n"
                output.write(text)

    # after all records processed
    with open(work_dir/'data'/'RB_Section_Summary.txt','w') as fp:
        for lionc in lionc_snomed_count_record:
            words = lionc_words_record[lionc]
            chars = lionc_characters_record[lionc]
            scts = lionc_snomed_count_record[lionc]
            rxnorms = lionc_rxnorm_count_record[lionc]
            line1 = "%10s" * 5 % (str(lionc), 'words', 'chars', '#snomed', '#rxnorm')
            line2 = ("%10s" + "%10.3f" * 4) % ('', mean(words), mean(chars), mean(scts), mean(rxnorms))
            print(line1)
            print(line2)
            fp.write(line1 + '\n')
            fp.write(line2 +'\n')


    # save found terms
    save_to_json(sct_to_desc, work_dir/'data'/'snomed_found.json', indent=4)
    save_to_json(rxcui_to_desc, work_dir/'data'/'rxcui_found.json', indent=4)


def find_full_codes(var, skip_key=None):
    term = "coding"
    if hasattr(var, 'items'):
        for key, value in var.items():
            if key == skip_key:
                continue
            if key == term:
                yield BasicCode({'coding': var['coding'], 'text': var['text']})
            if isinstance(value, dict):
                for result in find_full_codes(value):
                    yield result
            elif isinstance(value, list):
                for dict_ in value:
                    for result in find_full_codes(dict_):
                        yield result


class BasicCode():
    def __init__(self, FHIR_code):
        try:
            self.code_text = FHIR_code['text']
            if len(self.code_text.splitlines()) > 1:
                self.code_text = " ".join(self.code_text.split())
        except KeyError:
            self.code_text = ''

        try:  # sometimes code is not included and only text is
            self.code = FHIR_code['coding'][0]['code']
            system_text = FHIR_code['coding'][0]['system']
        except KeyError:
            self.code = '0'  # unknown code
            self.code_system = CodeSystem.OTHER
            return

        if system_text == SNOMED_REFERENCE:
            self.code_system = CodeSystem.SNOMED
        elif system_text == RXNORM_REFERENCE:
            self.code_system = CodeSystem.RXNORM
        else:
            self.code_system = CodeSystem.OTHER

    def __str__(self):
        return '(' + str(self.code) + ', "' + str(self.code_text) + '", ' + str(self.code_system) + ')'

    def __repr__(self):
        return self.__str__()

# negation status location might change
class BasicEntry():
    def __init__(self, FHIR_entry):
        self.FHIR_entry = FHIR_entry
        self.uuid = FHIR_entry['id']
        self.negation_status = NEGATION_CLAUSE in FHIR_entry  #check for NEGATION_CLAUSE
        #self.uncertainty_status = UNCERTAINTY_CLAUSE IN FHIR_entry

    def find_additional_codes(FHIR_entry, skip_key=None):
        additional_codes = list(find_full_codes(FHIR_entry, skip_key=skip_key))
        return additional_codes

    def return_codes(self, incl_addtl_codes = True):
        list_ = []
        if hasattr(self, 'main_code'):
            list_.append(self.main_code)
        if incl_addtl_codes and hasattr(self, 'additional_codes'):
            list_ += self.additional_codes
        return list_

    # return tuple of (#sct, #rxnorm) codes contained.
    def code_type_counts(self):
        snomed_count, rxcui_count = 0, 0
        list_ = []

        if hasattr(self, 'main_code'):
            list_.append(self.main_code.code_system)
        if hasattr(self, 'additional_codes'):
            for code in self.additional_codes:
                list_.append(code.code_system)

        for i in list_:
            if i == CodeSystem.SNOMED:
                snomed_count += 1
            elif i == CodeSystem.RXNORM:
                rxcui_count += 1
        return (snomed_count, rxcui_count)

# For each kind of entry, define where to find the main code and additional codes
class ConditionEntry(BasicEntry):
    def __init__(self, FHIR_entry):
        main_code_key = 'code'
        BasicEntry.__init__(self, FHIR_entry)
        self.main_code = BasicCode(FHIR_entry[main_code_key])
        self.type_ = self.main_code.code_system
        self.additional_codes = BasicEntry.find_additional_codes(FHIR_entry, skip_key=main_code_key)


class FamilyHistoryEntry(BasicEntry):
    def __init__(self, FHIR_entry):
        BasicEntry.__init__(self, FHIR_entry)
        self.main_code = BasicCode(FHIR_entry['condition'][0]['code'])
        self.type_ = self.main_code.code_system
        FHIR_copy = dict(FHIR_entry)
        FHIR_copy['condition'][0].pop('code', None)
        self.find_additional_codes(FHIR_copy)


class MedicationEntry(BasicEntry):
    def __init__(self, FHIR_entry):
        main_code_key = 'code'
        BasicEntry.__init__(self, FHIR_entry)
        self.main_code = BasicCode(FHIR_entry[main_code_key])
        self.type_ = self.main_code.code_system
        self.additional_codes = BasicEntry.find_additional_codes(FHIR_entry, skip_key=main_code_key)


class MedicationStatementEntry(BasicEntry):
    def __init__(self, FHIR_entry):
        main_code_key = 'medicationCodeableConcept'
        BasicEntry.__init__(self, FHIR_entry)
        self.main_code = BasicCode(FHIR_entry['medicationCodeableConcept'])
        self.type_ = self.main_code.code_system
        self.additional_codes = BasicEntry.find_additional_codes(FHIR_entry, skip_key=main_code_key)


class ProcedureEntry(BasicEntry):
    def __init__(self, FHIR_entry):
        main_code_key = 'code'
        BasicEntry.__init__(self, FHIR_entry)
        self.main_code = BasicCode(FHIR_entry[main_code_key])
        self.type_ = self.main_code.code_system
        self.additional_codes = BasicEntry.find_additional_codes(FHIR_entry, skip_key=main_code_key)


    # take entry and convert it into a list of codes using the following representation:
    # [lionc section, family history, code type, code, negation term] =>
    # lionc_(F-)(sct_)code(n)
    #
    # Also add to a list of codes if given a dictionary to keep track of the codes.
    # the list of codes from the dictionary can be later used for searching through ontologies
def entry_to_codes(entry: BasicEntry, uuid_to_section={}, incl_code_type=False, sct_to_desc=None,
                   rxcui_to_desc=None, incl_addtl_codes = True) -> (list, bool):
    negation = entry.negation_status
    #uncertainty = entry.uncertainty_status
    code_list = []
    is_family_history = isinstance(entry, FamilyHistoryEntry)

    section = find_section_for_uuid(entry.uuid, uuid_to_section) + '_'

    for full_code in entry.return_codes(incl_addtl_codes= incl_addtl_codes):
        code = full_code.code
        if code == '0':  # skip unknown codes
            continue
        text = full_code.code_text
        if full_code.code_system == CodeSystem.SNOMED:
            type_ = 'F-' * is_family_history + 'sct_' * incl_code_type
            if sct_to_desc is not None:
                sct_to_desc[code] = full_code.code_text
        elif full_code.code_system == CodeSystem.RXNORM:
            type_ = 'F-' * is_family_history + 'rxn_' * incl_code_type
            if rxcui_to_desc is not None:
                rxcui_to_desc[code] = full_code.code_text
        else:
            # do not add other codes
            continue
            # type_ = 'F-' * is_family_history

        new_code = section + type_ + code
        code_list.append(new_code)
    return code_list, negation


def find_section_for_uuid(uuid, uuid_to_section={}):
    if uuid in uuid_to_section:
        section = uuid_to_section[uuid]
    else:
        section = '00000-0'
    return section

def text_word_counter(line):
    _, body = clean_div(line)
    no_punctuation = str.maketrans(' ', ' ', string.punctuation)

    char_count = len(body)
    word_count = len(body.translate(no_punctuation).split())

    return word_count, char_count

def clean_div(div_text) -> (str,str):
    div_text = div_text.strip()
    nodiv = div_text[div_text.find('>') + 1:div_text.rfind('<')]

    title = nodiv[:find_first_punc(nodiv)]  # title of our section
    body = nodiv[find_first_punc(nodiv) + 1:]  # body of section

    body = re.sub(r"\s+", " ", body)
    body = remove_nl(body)

    return title, body

def find_first_punc(line):
    min_ = len(line)
    for x in string.punctuation:
        pos_ = line.find(x)
        if pos_> 0 and pos_ < min_:
            min_ = pos_
    return min_


def remove_nl(line):
    for i in range(1,len(line)):
        if line[i:i+2] == r'\n':
            return line[:i] + ' ' + remove_nl(line[i+2:])
        elif line[i] == '/' or line == '\\':
            return line[:i] + ' ' + remove_nl(line[i+1:])

    # else if none return line
    return line


class CodeSystem(Enum):
    SNOMED = 0
    RXNORM = 1
    OTHER = 2


if __name__ == "__main__":
    main()
