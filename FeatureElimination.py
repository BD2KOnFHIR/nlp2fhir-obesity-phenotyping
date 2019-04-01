#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FeatureElimination.py

Eliminates features from classifers using RFECV and performs classification
"""

import logging
from pathlib import Path
from collections import defaultdict
from MLDataProcessing import combine_list_dfs, save_to_json
from itertools import combinations
from RunClassification import rfe_classifier, rfecv_classifier, set_up_classifier
import pandas as pd
from MLDataProcessing import get_ML_parameters, rearrange_for_testing, log_settings, normalize_df_columns
import CalculatePerformance


def main(work_dir=None, model='rf', set_of_classes=(0, 1, 2, 3)):
    while work_dir is None or Path(work_dir).exists() is False:
        print("Unable to locate directory.")
        work_dir = input("Please enter working directory: ")

    # folder with features split by section
    work_dir = Path(work_dir)
    DATA_DIR = work_dir/'section_fm'
    GOLD_FILE = work_dir/'GOLD_multiclass.csv'




    ML_param_file = work_dir / 'data' / 'ML_model_settings' / 'ML_default_settings.json'
    if ML_param_file.exists():
        params = get_ML_parameters(use_default=False, dict_path=ML_param_file)
    else:
        params = get_ML_parameters(use_default=True)


    logging.info("Loading Data from: " + str(DATA_DIR))

    pathlist = Path(DATA_DIR).glob('*.csv')

    fm_by_section = {}
    lionc = []
    sections_writen = defaultdict(bool) # default = false

    for path in pathlist:
        section_name = path.stem
        lionc.append(section_name)
        fm_by_section[section_name] = pd.read_csv(path,index_col=0)
        fm_by_section[section_name].fillna(0, inplace=True)

    if len(lionc) < 1:
        logging.error("No files found at: " + str(DATA_DIR))
        exit()


    # load gold
    gold = pd.read_csv(GOLD_FILE,index_col=0)
    gold.fillna(0, inplace=True)
    tasks = [x for x in gold if x not in ['test','train']]

    frac_features_for_running_f1 = 0.01


    #set the following to use either RFECV or RFE
    run_f1_with_rfecv = True
    logging.info("frac_features_for_running_f1: " + str(frac_features_for_running_f1) + " with CV?: " + str(run_f1_with_rfecv))
    no_feature_elim = False  # if run_f1_with_rfecv == False can try to run without Feature elim

    logging.info("list of sections found:")
    logging.info(str(lionc))
    logging.info("model to run: " + str(model))

    rfecv_top_features = {}
    NUM_SECT_TO_COMBINE = len(lionc)  # add all sections
    sect_combinations = combinations(lionc,NUM_SECT_TO_COMBINE)

    for combo in sect_combinations:

        section_list = []
        for section in combo:
            section_list.append(fm_by_section[section])

        merged = combine_list_dfs(section_list)

        merged = normalize_df_columns(merged,0,tf=(lambda x: x ** (1/3)))

        train, test, features = rearrange_for_testing(merged, gold)

        p_avg, r_avg, f1_avg, f1_macro_avg = 0, 0, 0, 0

        print("features:", len(features))
        output_label_line = '%s %8s %8s %8s %8s %8s %8s' % ("Morbidity Results", "P-micro", "P-macro", "R-micro", "R-macro", "F1-micro", "F1-macro")
        logging.info(output_label_line)



        for task in tasks:
            train, test, features = rearrange_for_testing(merged, gold, task, set_of_classes)
            # filter features if desired
            features = [f for f in features if len(f)!=2]
            # features = [f for f in features if f[-1] != 'n']

            if run_f1_with_rfecv:
                preds, feat_important,num_feat = rfecv_classifier(model, train_data=train[features], train_class=train[task], test_data=test[features], CV_=3, fraction_feat_to_keep=frac_features_for_running_f1, LM_params=params, save_model=True)
                rfecv_top_features[task] = feat_important
            elif no_feature_elim:
                clf = set_up_classifier(model, 0, LM_params=params)
                clf.fit(train[features], train[task])
                preds = clf.predict(test[features])
            else:
                preds, feat_important,num_feat = rfe_classifier(model, train_data=train[features], train_class=train[task].astype(int), test_data=test[features], CV_=10, fraction_feat_to_keep=frac_features_for_running_f1, LM_params=params)




            results = CalculatePerformance.calculate_metrics(list(test[task]), list(preds), set_of_classes,output_type='values')
            f1 = results[4]
            f1_macro = results[5]

            f1_avg += f1 /len(tasks)
            f1_macro_avg += f1_macro/len(tasks)


            results = CalculatePerformance.calculate_metrics(list(test[task]), list(preds), set_of_classes,output_type='values')
            logging.info("task: " + str(task) + ' ' + CalculatePerformance.calculate_metrics(list(test[task]), list(preds),set_of_classes,output_type='text').strip())





    file_name = work_dir / 'models' / 'top_features.json'
    save_to_json(rfecv_top_features,file_name)

    logging.info("Averages: f1: %.6f, f1_macro: %.6f" % (f1_avg, f1_macro_avg))


if __name__ == '__main__':
    log_settings(filename="FeatureElimination.log")
    main(work_dir=None, model='dt')