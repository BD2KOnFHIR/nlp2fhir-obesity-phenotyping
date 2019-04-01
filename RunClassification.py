#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RunClassification.py

Original script for running classification tasks from one feature matrix.  Still in prototyping stage and might require manually
changing hardcoded settings.
"""






import numpy as np
import pandas as pd
import sklearn
import CalculatePerformance
import logging

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from random import random
from datetime import datetime
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from collections import Counter
from math import isnan
from warnings import warn
from MLDataProcessing import myFactorize, get_ML_parameters, generate_param_strings, log_settings, rearrange_for_testing, normalize_df_columns

import time

have_written_params_to_file = False



def run_rfe_classifier(method, train_data, train_class, test_data, CV_ = 0, fraction_feat_to_keep = 0.1, LM_params = get_ML_parameters()):
    global have_written_params_to_file
    if have_written_params_to_file is False:
        logging.info("Run settings for models:")
        logging.info(str(LM_params))
        have_written_params_to_file = True

    clf = set_up_classifier(method, CV_, LM_params)

    if CV_ < 1 and method != 'svm':
        clf = OneVsRestClassifier(clf)


    # fit and predict based on whether cross validation is used
    if (CV_ > 1):
        step_elim = (1-fraction_feat_to_keep)/CV_
        rfecv = RFE(estimator=clf, step=step_elim, n_features_to_select=int(fraction_feat_to_keep * len(list(train_data))))
        rfecv.fit(train_data, train_class)
        preds = rfecv.predict(test_data)
    else:
        clf.fit(train_data, train_class)
        preds = clf.predict(test_data)

    return preds

# have to find index from test for which we add predictions.  Some index are not added since the gold file did not have
# a valid class (i.e. blank instead of 'Y'/'N')
def append_results_to_df(base_df, test, preds, task):
    task_results = str(task) + ' results'
    results_row_df = pd.DataFrame(preds, index=test.index, columns=[task_results])
    base_df = pd.concat([base_df, results_row_df], axis=1)
    base_df[task_results].fillna(-1, inplace=True)
    return base_df


def eval_classifier(method, train_data, train_class, test_data, test_class, LM_params = get_ML_parameters(), positive_roc_index = 1):
    global have_written_params_to_file
    if have_written_params_to_file is False:
        logging.info("Run settings for models:")
        logging.info(str(LM_params))
        logging.info("First run method: " + str(method))
        have_written_params_to_file = True

    # set classifier method
    if method =='svm':
        clf = SVC(random_state=0, probability=True, **LM_params['svm'])
    else:
        clf = set_up_classifier(method, 0, LM_params)

    clf = OneVsRestClassifier(clf)

    clf.fit(train_data, train_class)
    probas_ = clf.predict_proba(test_data)
    preds = clf.predict(test_data)
    fpr, tpr, thresholds = roc_curve(test_class, probas_[:, positive_roc_index], pos_label=positive_roc_index)
    roc_auc = auc(fpr, tpr)
    roc_auc = max(roc_auc, 1 - roc_auc)

    return preds, roc_auc

# run recursive feature elimination
def rfe_classifier(method, train_data, train_class, test_data, CV_ = 3, fraction_feat_to_keep = 0.1, LM_params = get_ML_parameters()):
    global have_written_params_to_file
    if have_written_params_to_file is False:
        logging.info("Run settings for models:")
        logging.info(str(LM_params))
        have_written_params_to_file = True

    clf = set_up_classifier(method, CV_, LM_params)

    # fit and predict based on whether cross validation is used
    if (CV_ > 1):
        step_elim = (1-fraction_feat_to_keep)/CV_
        num_to_keep = int(fraction_feat_to_keep * len(list(train_data)))
        num_to_keep = max(num_to_keep, 1)
        rfecv = RFE(estimator=clf, step=step_elim, n_features_to_select=num_to_keep)

        rfecv.fit(train_data, train_class)
        preds = rfecv.predict(test_data)
        mask = list(rfecv.support_)
        # print("Number of features selected:", sum(mask))
        #print(rfecv.ranking_)
        features = train_data.columns
        features_selected = [features[i] for i in range(0, len(mask)) if mask[i]]
        #print(features_selected)

    else:
        clf.fit(train_data, train_class)
        preds = clf.predict(test_data)

    return preds, features_selected, sum(mask)

def set_up_classifier(method, CV_, LM_params):
    if method == 'dt':
        clf = tree.DecisionTreeClassifier(random_state=0, **LM_params['dt'])
    elif method =='rf':
        clf = RandomForestClassifier(random_state=0,**LM_params['rf'])
    elif method == 'lr':
        clf = LogisticRegression(random_state = 0,**LM_params['lr'])
    elif method =='svm':
        if CV_ > 1:
            if LM_params['svm']['kernel'] != 'linear':
                logging.warn("SVM kernel method set to linear to do cross validation")
                LM_params['svm']['kernel'] = 'linear'
        clf = SVC(random_state=0, **LM_params['svm'])
    elif method =='gb':
        clf = GradientBoostingClassifier(random_state=0, **LM_params['gb'])
    elif method =='nb':
        clf = MultinomialNB(**LM_params['nb'])
    else:
        warn("Invalid method selected running DT as default method")
        clf = tree.DecisionTreeClassifier(random_state=0, **LM_params['dt'])
    return clf

#run recursive feature elimination with cross validation
def rfecv_classifier(method, train_data, train_class, test_data, CV_ = 3, fraction_feat_to_keep = 0.1, LM_params = get_ML_parameters(), save_model=False):
    n_orig_features = len(list(train_data))
    max_ratio_diff = 1.2
    global have_written_params_to_file
    if have_written_params_to_file is False:
        logging.info("Run settings for models:")
        logging.info(str(LM_params))
        have_written_params_to_file = True
    # set classifier method

    clf = set_up_classifier(method, CV_, LM_params)

    # fit and predict based on whether cross validation is used
    if (CV_ > 1):
        step_elim = (1-fraction_feat_to_keep)/CV_

        # Recursive feature elimination with Cross Validation
        # CV might have issues if data set classification is poorly balanced and can not split it properly
        try:
            rfecv = RFECV(estimator=clf, step=step_elim, cv=StratifiedKFold(n_splits=CV_, random_state=0),
                          scoring='accuracy')
            rfecv.fit(train_data, train_class)
            preds = rfecv.predict(test_data)

            current_fraction_features = rfecv.n_features_ / n_orig_features
            if (current_fraction_features * max_ratio_diff < fraction_feat_to_keep):
                raise ValueError("Not enough features kept by RFECV defaulting to RFE")
        except ValueError:
            rfecv = RFE(estimator=clf, step=step_elim, n_features_to_select=int(fraction_feat_to_keep * len(list(train_data))))
            rfecv.fit(train_data, train_class)
            preds = rfecv.predict(test_data)

        mask = list(rfecv.support_)
        features = train_data.columns
        features_selected = [features[i] for i in range(0, len(mask)) if mask[i]]

        # sometimes RFECV does not eliminate enough features, so then lets run RFE to remove more if more than 20% over
        current_fraction_features = len(features_selected)/n_orig_features
        step_elim = ( current_fraction_features - fraction_feat_to_keep) / CV_
        if (current_fraction_features > max_ratio_diff * fraction_feat_to_keep) and step_elim > 0:
            rfecv = RFE(estimator=clf, step=step_elim, n_features_to_select=int(fraction_feat_to_keep * n_orig_features))
            rfecv.fit(train_data[features_selected], train_class)
            preds = rfecv.predict(test_data[features_selected])
            mask = list(rfecv.support_)
            features = train_data.columns
            features_selected = [features[i] for i in range(0, len(mask)) if mask[i]]

    else:
        clf.fit(train_data, train_class)
        preds = clf.predict(test_data)

    return preds, features_selected, sum(mask)



if __name__ == "__main__":

    log_settings(filename="RunClassification.log", level=logging.INFO)

    logging.info("pandas version " + str(pd.__version__))
    logging.info("sklearn version " + str(sklearn.__version__))
    logging.info("numpy version " + str(np.__version__))
