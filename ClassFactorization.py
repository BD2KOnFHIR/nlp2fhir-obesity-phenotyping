#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ClassFactorization.py

Using a user defined dictionary, will convert text labels of a class into an "int" value for more convenience
when running ML algorithms.  e.g. 'Y' to 1, 'N' to 0
"""


import pandas as pd
from pathlib import Path

protected_key_words = ['train', 'test']
conversion = {'Y': 1, 'N': 0, 'Q': 2, 'U': 3}



def _data_multi_factorize(*args):
    """Converts text labels in gold standard to integers based on global conversion dictionary
    e.g. 'Y' to 1
    Converts anything that is not an integer and not in dictionary to -1 (which will be ignored by classifers later on)
    """
    x = args[0]
    global conversion, protected_key_words

    if x in conversion:
        return conversion[x]
    elif isinstance(x, int):
        pass
    else:
        return -1



def main(gold_csv = None, conversion_dict = None, work_dir = None):
    while gold_csv is None or Path(gold_csv).exists() is False:
        gold_csv = input("Please enter path for gold csv or q to quit:")
        if gold_csv == 'q':
            exit()

    gold_csv = Path(gold_csv)

    if conversion_dict is not None:
        global conversion
        conversion = conversion_dict

    df = pd.read_csv(gold_csv, index_col=0)

    print("Columns: ")
    print(list(df.columns))

    for task in df.columns:
        if task in protected_key_words:
            continue
        df[task] = df[task].apply(_data_multi_factorize)
    print(df.head())

    if work_dir is None:
        path_out = gold_csv.parent / (gold_csv.stem + '_multiclass.csv')
    else:
        path_out= Path(work_dir) / (gold_csv.stem + '_multiclass.csv')
    df.to_csv(path_out)

    print("Factorized Gold Standard saved to: ", str(path_out))

if __name__ == '__main__':
    main()
