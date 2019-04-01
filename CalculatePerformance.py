#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CalculatePerformance.py

Methods included for estimating the performance of classification predictions.  Mainly F1 scores.

Main method is out-dated (originally set-up to evaluate RunClassifier output results).
"""



#ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path

work_dir = Path("")
GOLD_FILE = work_dir / "GOLD_multiclass.csv"


def binary_evaluation(gold, pred, ok_set = (0,1), default_value = None):
    valid_preds = [pred[i] for i in range(len(gold)) if (gold[i] in ok_set)]
    valid_gold = [gold[i] for i in range(len(gold)) if (gold[i] in ok_set)]

    if default_value is None: # default for Negative Results
        default_value = min(ok_set) # might need to find better solution
    for i in range(len(valid_preds)):
        if valid_preds[i] not in ok_set:
            valid_preds[i] = default_value
        if valid_gold[i] not in ok_set:
            valid_gold[i] = default_value

    p_ = precision_score(valid_gold, valid_preds)
    r_ = recall_score(valid_gold, valid_preds)
    f1_ = f1_score(valid_gold, valid_preds)

    return (p_,r_,f1_)

def simple_f1(gold, pred, acceptable_codes = ['Y','Q','N','U']):
    acceptable_codes = list(acceptable_codes)
    valid_preds = [pred[i] for i in range(len(gold)) if gold[i] in acceptable_codes]
    valid_gold = [gold[i] for i in range(len(gold)) if gold[i] in acceptable_codes]

    if len(set(valid_gold)) == 2:
        f1 = f1_score(valid_gold, valid_preds)
    else:
        f1 = f1_score(valid_gold, valid_preds, average='micro')
    return f1


def filter_valid(gold, pred, acceptable_codes = None):
    gold = list(gold)
    pred = list(pred)
    valid_gold = []
    valid_pred = []
    if acceptable_codes != None:
        for i in range(0,len(gold)):
            if gold[i] in acceptable_codes: # ignore blank or non conformant data (if they are not in acceptable_codes)
                valid_gold.append(gold[i])
                valid_pred.append(pred[i])
        return valid_gold, valid_pred
    else:
        return gold, pred


def calculate_metrics(gold, pred, acceptable_codes = None, output_type='text'):
    results = []
    gold = list(gold)
    pred = list(pred)

    valid_gold, valid_pred = filter_valid(gold, pred, acceptable_codes=acceptable_codes)

    r_micro = recall_score(valid_gold, valid_pred, average='micro')
    f1_micro = f1_score(valid_gold, valid_pred, average='micro')
    p_micro = precision_score(valid_gold, valid_pred, average='micro')

    f1_macro = f1_score(valid_gold, valid_pred, average='macro')
    p_macro = precision_score(valid_gold, valid_pred, average='macro')
    r_macro = recall_score(valid_gold, valid_pred, average='macro')

    results = [p_micro, p_macro,r_micro, r_macro, f1_micro,f1_macro]

    if output_type == 'text':
        return (' %8.4f'*6 + '\n') % (p_micro, p_macro,r_micro, r_macro, f1_micro,f1_macro)
    elif output_type == 'tuple':
        return tuple(results)
    else:
        return results
