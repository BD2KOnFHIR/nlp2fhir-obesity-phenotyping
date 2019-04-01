#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MLDataProcessing.py

Misc functions to help run ML models

"""

import json
import os
import pickle
import logging
from pathlib import Path

from collections import Counter
from numpy import isnan
import pandas as pd


ML_settings_location = None
lionc_to_description = {'00000-0':'Unknown', '10155-0':'Allergies', '10157-6':'Family History', '10160-0':'Medication', '10164-2':'History of present illness', '10188-1':'General Overview', '11320-9':'Diet', '11330-8':'Alcohol use', '11366-2':'Tobaco use', '11450-4':'Problem List', '11451-2':'Psychiatric','29299-5':'Chief Complaint','29545-1':'Physical Exam','29546-9':'Review of Symptoms', '29762-2':'Personal/Social History', '47519-4':'Past Surgical History', '11338-1':'Past medical History'}


def rearrange_for_testing(fm, gold, task = None, set_of_classes = None):
    features = fm.columns
    full = pd.concat([fm, gold], axis=1)
    train, test = full[full['train'] == 1], full[full['test'] == 1]
    if set_of_classes != None and task != None:
        train = train[train[task].isin(set_of_classes)]  # remove classes not in set of classes allowed (e.g. blank)
        test = test[test[task].isin(set_of_classes)]
    return train, test, features


def lionc_list_to_description(lionc):
    output = ''
    for section in lionc:
        if section in lionc_to_description.keys():
            output = output + str(lionc_to_description[section]) + ', '
        else:
            output = output + "Unknown Section" + ', '

    return output[:-2]

def myFactorize(data, min_count=2, max_categories=None):   # min_count is miniumum required entries for it to be classified as valid and included

    accepted_categories = []
    counted = Counter(data)
    num_categories = 0

    for key, value in counted.most_common():
        if max_categories is not None and (num_categories >= max_categories):
            break

        if isinstance(key, float):
            if isnan(key):
                continue    #skip if key counted is not a number
        if value >= min_count:
            # print(key, " was accepted as key")
            accepted_categories.append(key)
            num_categories += 1
        else:
            break

    cat = pd.Categorical(data, categories=accepted_categories)

    return pd.factorize(cat, sort=True)

def generate_weighting(handle,weight_desired):
    # example weight:
    # INT_weight_desired = {'N': 100, 'Y': 100, 'Q': 0, 'U': 0}  # int Y/N
    # TEX_weight_desired = {'N': 0, 'Y': 100, 'Q': 0, 'U': 100}  # txt Y/U

    # need to find corresponding handles and weigh them
    weights = {}
    for key in range(0, len(handle)):
        weights[key] = weight_desired[handle[key]]



def generate_param_strings(params) -> dict:
    output = {}
    for model in params.keys():
        model_string = ''
        for key, value in params[model].items():
            model_string += '_' + str(key) + '=' + str(value)
            output[model] = model_string
    return output


def get_ML_parameters(use_default = True, dict_path = None)->dict:
    global ML_settings_location
    if use_default:
        if dict_path is not None:
            try:
                params = load_dict_json(dict_path, create_local_if_not_found=False)
            except Exception:
                pass

        if ML_settings_location is not None:
            try:
                params = load_dict_json(ML_settings_location, create_local_if_not_found=False)
            except Exception:
                pass
        # return default parameters
        return generate_default_ML_parameters()


    if dict_path is None and ML_settings_location is not None:
        dict_path = ML_settings_location

    if use_default is False and dict_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = Path(filedialog.askopenfilename(title="Select ML Parameters File"))
        params = load_dict_json(file_path, create_local_if_not_found=False)
        ML_settings_location = file_path
        root.quit()
        return params

    if use_default is False and dict_path is not None:
        params = load_dict_json(dict_path, create_local_if_not_found=False)
        return params


def generate_default_ML_parameters()->dict:
    dt_params = {
        'criterion': 'gini',
        'max_depth': 9,
        'max_leaf_nodes': 100,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'min_impurity_decrease': 0.006
    }
    rf_params = {
        'criterion': 'gini',
        'n_estimators': 80,
        'n_jobs': 1,
        'min_impurity_decrease': 0.001
    }
    svm_params = {
        'C': 100,
        'kernel': 'rbf'
    }
    lr_params = {
        'penalty': 'l2',
        'C': 10,
        'solver': 'liblinear',
        'n_jobs': 1
    }
    gb_params = {
        'learning_rate': 0.025,
        'n_estimators': 50,
        'max_depth': 3
    }

    nb_params = {}
    params = {'dt': dt_params, 'rf': rf_params, 'lr': lr_params, 'svm': svm_params, 'gb': gb_params, 'nb': nb_params}

    return params



# tools to combine feature sets
def combine_two_dfs(df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
    set1 = set(df1)
    set2 = set(df2)
    set_union = set1.intersection(set2)
    set2_uniques = set2.difference(set1)

    df_out = df1.copy()
    df_out.loc[:, set_union] += df2.loc[:, set_union]   # add missing values
    df_out = pd.concat([df_out, df2.loc[:, set2_uniques]], axis='columns') #append set2 data
    return df_out


def combine_list_dfs(list_of_df:list) ->pd.DataFrame:
    while len(list_of_df) > 1:
        list_of_df[0] = combine_two_dfs(list_of_df[0], list_of_df.pop())
    return list_of_df[0]


def combine_from_indices(dict_of_sections, selected_indices) ->pd.DataFrame:
    list_to_run = []
    for section in selected_indices:
        print(section)
        list_to_run.append(dict_of_sections[section])
    return combine_list_dfs(list_to_run)


    # if no transformation is specified it will default to 'one hot encoding' style 'binary' approach where the value
    # is either 1 or 0
def normalize_df_columns(df1, start_col:int = 0, tf= None)->pd.DataFrame:
    num_columns = len(list(df1))
    if start_col >= num_columns or start_col < 0:
        print("Attempting to normalize dataframe with start_col outside of index")
        return df1

    if tf is None:
        tf = (lambda x: 1 if x > 0 else 0)
        df1 = df1.applymap(tf)
    else:
        df1 = df1.applymap(tf)
        df1.fillna(0)

        from sklearn import preprocessing

        x = df1.values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df1 = pd.DataFrame(x_scaled, columns=df1.columns, index = df1.index)

    return df1


def save_to_json(obj_to_save, filename, indent = None, print_save_loc = False):
    if len(os.path.dirname(filename)) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fp:
        json.dump(obj_to_save, fp, indent=indent)
        if print_save_loc:
            print("Saving data to: ", str(filename))


def load_dict_json(filename, create_local_if_not_found = False):
    try:
        with open(filename, 'r') as fp:
            json_str = fp.read()
            result = json.loads(json_str)
    except FileNotFoundError as e:
        if create_local_if_not_found:
            print("Could not find file: ", str(filename), " creating new dictionary")
            result = {}
        else:
            raise e
    return result


def load_dict_pickle(filename):
    if filename.find('.') <= 0:
        filename = filename + '.p'
    try:
        open_file = open(filename, 'rb')
        object_returned = pickle.load(open_file)
        open_file.close()
    except Exception as e:
        print("Issue encountered reading file: ", str(filename))
        print(type(e))
        print("creating blank dictionary")
        object_returned = {}
    return object_returned


def pickle_something(obj_to_save, filename):
    filename = str(filename)
    if filename.find('.') <= 0:
        filename = filename + '.p'
    try:
        open_file = open(filename, 'wb')
        pickle.dump(obj_to_save, open_file, 0)
        open_file.close()
    except IOError as e:
        print(type(e))
        print(e)


def log_settings(filename = "Default.log", level=logging.INFO, filemode='a', stdout=True):
    import sys

    logging.basicConfig(filename=filename, level=level, filemode=filemode)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if stdout:
        #check to see if there is already a stdout handler
        for handler in root.handlers:
            if type(handler) == logging.StreamHandler:
                return
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        root.addHandler(ch)



def save_default_ML_params(work_dir = None, overwrite = False):
    from pathlib import Path
    while work_dir is None or Path(work_dir).exists() is False:
        work_dir = input("Please enter working directory: ")
    work_dir = Path(work_dir)
    file_path = work_dir / 'data' / 'ML_model_settings'/'ML_default_settings.json'

    global ML_settings_location
    ML_settings_location = file_path

    if file_path.exists() and overwrite is False:
        try:
            _ = load_dict_json(file_path, create_local_if_not_found=False)
            logging.info("ML_parameters already exists at: " + str(file_path))
            logging.info("ML file not overwritten.")
            return
        except Exception:
            pass

    save_to_json(obj_to_save=generate_default_ML_parameters(), filename=file_path, indent=4)


def save_df(df:pd.DataFrame, file):
    df.mask(df.eq(0)).to_csv(file)

def load_df(file):
    df = pd.read_csv(file,index_col=0)
    df.fillna(0, inplace=True)
    return df

if __name__ == '__main__':
    save_default_ML_params('C:/')