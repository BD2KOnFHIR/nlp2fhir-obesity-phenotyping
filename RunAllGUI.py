#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RunAllGUI.py

Runs all the pre-proccessing tasks to convert FHIR json ResourceBundle Reports into a CSV feature matrix.

Runs JsonBasedReader to search for Rxcui and Snomed CT codes.
Ontology search is run using SnomedOntologyLookup, RxOntologyLookup.
AggregateReportsBySection creates a CSV feature matrix from the previous outputs for running ML tasks.
ClassFactorization is a simple script that converts text ('Y','N') to int values (1,0)

The "Gold File" is also expected to have train and test columns with 1's to indicate that they should be assigned
to corresponding train or test groups.

RunClassification, FeatureElimination might require manual tweaking or it might be better to start over and write your own code to run the classification
"""





import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import shutil


DATA_DIR = ''
WORK_DIR = ''
GOLD_CSV = ''

add_snomed_ontology = True
snomed_ontology_ancestor_lookup_depth = 2
add_rxnorm_ATC = True
convert_rxcui_to_ingred = True
keep_rxnorm_after_conversion = True
gold_factorization = "{'Y': 1, 'N': 0, 'Q': 2, 'U': 3}"



master = tk.Tk()
master.title("FHIR2ML JSON Resource Bundle to Snomed/RxNorm CSV")



def run_tasks():
    global add_snomed_ontology, snomed_ontology_ancestor_lookup_depth, add_rxnorm_ATC, convert_rxcui_to_ingred, keep_rxnorm_after_conversion
    global DATA_DIR, WORK_DIR, GOLD_CSV
    add_snomed_ontology = var_sno.get()
    add_rxnorm_ATC = var_atc.get()
    convert_rxcui_to_ingred = var_ing.get()
    keep_rxnorm_after_conversion = var_keep.get()
    DATA_DIR = str(e1.get())
    WORK_DIR = str(e2.get())
    GOLD_CSV = str(e3.get())
    snomed_ontology_ancestor_lookup_depth = int(e6.get())
    master.quit()


def load01():
    folderdialog_to_entry(e1, "Select FHIR JSON Resource Bundle Directory")

def load02():
    folderdialog_to_entry(e2, "Select Output or Working Directory")

def load03():
    filedialog_to_entry(e3, "Select Gold File")

def filedialog_to_entry(ex, description):
    file_ = filedialog.askopenfilename(title=description)
    if file_:
        ex.delete(0,tk.END)
        ex.insert(0,str(file_))
    return

def folderdialog_to_entry(ex, description):
    dir_ = filedialog.askdirectory(title=description)
    if dir_:
        ex.delete(0,tk.END)
        ex.insert(0,str(dir_))
    return

tk.Label(master, text="Resource Bundle Folder").grid(row=0, sticky=tk.W)
tk.Label(master, text="Output Folder").grid(row=1, sticky=tk.W)
tk.Label(master, text="Gold Standard").grid(row=2, sticky=tk.W)
tk.Label(master, text="Gold Class Factorization").grid(row=3, sticky=tk.W)
e1 = tk.Entry(master, width = 50)
e2 = tk.Entry(master, width = 50)
e3 = tk.Entry(master, width = 50)
e4 = tk.Entry(master, width = 50)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e4.insert(tk.E, gold_factorization)

tk.Label(master, text="Options:").grid(row=4, sticky=tk.W)
var_sno = tk.BooleanVar()
var_sno.set(value=add_snomed_ontology)
var_atc = tk.BooleanVar()
var_atc.set(value=add_rxnorm_ATC)
var_ing = tk.BooleanVar()
var_ing.set(value=convert_rxcui_to_ingred)
var_keep = tk.BooleanVar()
var_keep.set(value=keep_rxnorm_after_conversion)

var_depth = tk.IntVar()
var_depth.set(value=3)


tk.Checkbutton(master, text="Include Snomed Ontology", variable=var_sno).grid(row=5, sticky=tk.W)
tk.Checkbutton(master, text="Include RxNorm ATC Classification", variable=var_atc).grid(row=7, sticky=tk.W)
tk.Checkbutton(master, text="Include RxNorm Ingredients", variable=var_ing).grid(row=8, sticky=tk.W)
tk.Checkbutton(master, text="Keep orig. RxNorm if converted to ingredients", variable=var_keep).grid(row=9, sticky=tk.W)

l6 = tk.Label(master, text="Snomed Ancestor Lookup Depth").grid(row=6, sticky=tk.W)
e6 = tk.Entry(master, width = 5)
e6.grid(row=6, column=1, sticky=tk.W, padx=4, pady=4)
e6.insert(tk.E, "3")


tk.Button(master, text='Quit', command=master.quit).grid(row=10, sticky=tk.E, padx=4, pady=4)
tk.Button(master, text='Run', command=run_tasks).grid(row=10, sticky=tk.W, padx=4, pady=4)

b1 = tk.Button(master, text='Open',command=load01).grid(row=0,column=2, sticky=tk.W, padx=4, pady=2)
b2 = tk.Button(master, text='Open',command=load02).grid(row=1,column=2, sticky=tk.W, padx=4, pady=2)
b3 = tk.Button(master, text='Open',command=load03).grid(row=2,column=2, sticky=tk.W, padx=4, pady=2)

tk.mainloop()
master.quit()


if DATA_DIR and WORK_DIR and GOLD_CSV:
    import JsonBasedReader
    import AggregateReportsBySection
    import SnomedOntologyLookup
    import RxOntologyLookup
    import MLDataProcessing
    import ClassFactorization
    import FeatureElimination
    import ast
    gold_factorization = ast.literal_eval(gold_factorization)

    try:
        shutil.copytree('./data', Path(WORK_DIR) / 'data')
    except Exception:
        pass

    MLDataProcessing.save_default_ML_params(work_dir=WORK_DIR)  #
    JsonBasedReader.main(data_dir=DATA_DIR, work_dir=WORK_DIR)

    if add_snomed_ontology:
        SnomedOntologyLookup.main(WORK_DIR, depth= snomed_ontology_ancestor_lookup_depth)

    RxOntologyLookup.main(WORK_DIR, find_ATC=add_rxnorm_ATC, find_ingreds=add_rxnorm_ATC)

    disregard_negation_when_adding_original_codes = True
    AggregateReportsBySection.main(WORK_DIR, add_rxnorm_ATC=add_rxnorm_ATC, add_snomed_ontology=add_snomed_ontology,convert_rxcui_to_ingred=convert_rxcui_to_ingred, keep_rxnorm_after_conversion=keep_rxnorm_after_conversion, disregard_negation_when_adding_original_codes=disregard_negation_when_adding_original_codes)

    ClassFactorization.main(gold_csv=GOLD_CSV, conversion_dict= gold_factorization,work_dir=WORK_DIR)

    FeatureElimination.main(work_dir=WORK_DIR, model='rf', set_of_classes=set(gold_factorization.values()))

    print("Basic processing complete.")

