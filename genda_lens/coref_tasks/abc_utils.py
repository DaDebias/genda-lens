"""
Utility functions for running the ABC coreference task.

Builds on the codebase developed for the following paper: 

Title: “Type B reflexivization as an unambiguous testbed for multilingual multi-task gender bias.”
Authors: González, A. V., Barrett, M., Hvingelby, R., Webster, K., & Søgaard, A.
Date: 2020
Code availability: https://github.com/anavaleriagonzalez/ABC-dataset
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import fairlearn.metrics as flm
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.metrics import classification_report, f1_score
import progressbar


def run_abc_coref(coref_model):
    """
    Run the coreference model on the ABC dataset and return predictions. (Duration approx. 20 min.)
    """
    # load stereotypically male occupation sentences
    with open(
        os.path.join(Path(__file__).parents[1], "data", "abc_male_sents.txt"), "r"
    ) as f:
        male_abc = f.readlines()
    # load stereotypically female occupation sentences
    with open(
        os.path.join(Path(__file__).parents[1], "data", "abc_fem_sents.txt"), "r"
    ) as f:
        fem_abc = f.readlines()

    all_preds = []
    for gendered_data in [fem_abc, male_abc]:
        data_ = [line_ for line_ in gendered_data if line_ != "---\n"]
        i = 0
        preds = []

        # PROGRESS BAR
        abc_bar = progressbar.ProgressBar(maxval=len(data_)).start()

        # run the model on the data with the predict method and save the predictions
        for idx, i in enumerate(data_):
            line = [i.split()]
            try:
                # get preds
                predicted_clusters = coref_model.predict(line)
            except:
                print(line)
            preds.append(predicted_clusters)
            abc_bar.update(idx)
        abc_bar.finish()

        all_preds.append(preds)
    fem_preds = all_preds[0]
    male_preds = all_preds[1]

    return fem_preds, male_preds


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_coref_predictions(chunk_preds):
    """Get the coreference predictions for the reflexive case and the anti-reflexive cases.

    Args:
        - chunk_preds: list of predictions
        - coref_output_file: the name of the file where the predictions are stored

    Returns:
        - fem
        - male
        - reflexive
    """

    ref, male, fem = [], [], []

    # the truth labels for the reflexive case, i.e. all 1s
    labels_ref = list(np.repeat(1, len(chunk_preds)))
    # the truth labels for the anti-reflexive cases, i.e. all 0s
    labels_anti_ref = list(np.repeat(0, len(chunk_preds)))

    for p in chunk_preds:
        # assert if %3 == 0

        ref_pred, male_pred, fem_pred = p[0], p[1], p[2]

        predicted_clusters = "clusters"  # could be an input argument to the function to ensure that the correct key is used in case of tesitng pther models

        cluster_ref = ref_pred[predicted_clusters]
        cluster_male = male_pred[predicted_clusters]
        cluster_fem = fem_pred[predicted_clusters]
        clusters = [cluster_ref, cluster_male, cluster_fem]

        for i, cluster in enumerate(clusters):
            if i == 0:
                if cluster != []:
                    ref.append(1)
                else:
                    ref.append(0)
            elif i == 1:
                if cluster != []:
                    male.append(1)
                else:
                    male.append(0)
            elif i == 2:
                if cluster != []:
                    fem.append(1)
                else:
                    fem.append(0)
    return ref, fem, male, labels_ref, labels_anti_ref


def get_positive_preds(chunk_preds):
    """Get False Posivie Rates for the reflexive case and the anti-reflexive cases and print output tables."""

    ref, fem, male, labels_ref, labels_anti_ref = get_coref_predictions(chunk_preds)
    labels_preds_ = zip(
        [labels_ref, labels_anti_ref, labels_anti_ref], [ref, fem, male]
    )
    # the tags for the classification reports
    tags = ["reflexive", "anti_reflexive_female", "anti_reflexive_male"]
    res = {}
    for idx, (label, pred) in enumerate(labels_preds_):
        if label == labels_ref:
            # get the true positive rate for the current label and prediction
            res[f"tpr_{tags[idx]}"] = flm.true_positive_rate(label, pred, pos_label=1)

        elif label == labels_anti_ref:
            # get the false positive rate for the current label and prediction
            res[f"fpr_{tags[idx]}"] = flm.false_positive_rate(label, pred, pos_label=1)

    return res


def evaluate_coref_abc(fem_preds, male_preds):
    for gender in ["fem", "male", "all_sents"]:  # and for all
        if gender == "fem":
            preds = fem_preds
        elif gender == "male":
            preds = male_preds
        elif gender == "all_sents":
            preds = fem_preds + male_preds  # merge the two lists

        # split the predictions into chunks of 3
        chunk_preds = list(chunks(preds, 3))  # danish

        # get the false positive rates for the reflexive and anti-reflexive cases
        results = get_positive_preds(chunk_preds)

        # save the results in a dataframe
        if gender == "fem":
            df_fem = results
        elif gender == "male":
            df_male = results
        elif gender == "all_sents":
            df_all_sents = results

    return df_fem, df_male, df_all_sents  # dicts


def logratio(x: float, y: float):
    return round(math.log2(x / y), 2)


def eval_results(fem: dict, male: dict, all: dict, model_name: str):
    """Evaluate the results of the coreference model on the ABC data."""

    data = [
        fem["tpr_reflexive"],
        fem["fpr_anti_reflexive_female"],
        fem["fpr_anti_reflexive_male"],
        male["tpr_reflexive"],
        male["fpr_anti_reflexive_female"],
        male["fpr_anti_reflexive_male"],
        all["tpr_reflexive"],
        all["fpr_anti_reflexive_female"],
        all["fpr_anti_reflexive_male"],
    ]

    mean_refl = np.mean([fem["tpr_reflexive"], male["tpr_reflexive"]])
    mean_fem = np.mean(
        [fem["fpr_anti_reflexive_female"], male["fpr_anti_reflexive_female"]]
    )
    mean_male = np.mean(
        [fem["fpr_anti_reflexive_male"], male["fpr_anti_reflexive_male"]]
    )

    gender_effect_size = logratio(mean_male, mean_fem)

    data = [[round(data_, 2) for data_ in data]]

    simlpe_data = {
        f"Simple Output for {model_name}": ["", "ABC", ""],
        "Gender Effect Size": ["", gender_effect_size, ""],
        "Pronoun": ["Female", "Male", "Reflexive"],
        "Mean Rate of Detected Clusters": [mean_fem, mean_male, mean_refl],
    }

    simlpe_data = pd.DataFrame(simlpe_data).T

    # Set the first row as the column names
    simlpe_data.columns = simlpe_data.iloc[0]

    # Drop the first row
    simple_df = simlpe_data.iloc[1:]

    detailed_data = {
        f"Detailed Output for {model_name}": ["", "", "ABC", "", "", ""],
        "Gender Effect Size": ["", "", gender_effect_size, "", "", ""],
        "Pronoun": ["", "Female", "", "Male", "", "Reflexive"],
        "Mean Rate of Detected Clusters": ["", mean_fem, "", mean_male, "", mean_refl],
        "Stereotypical Occupation": [
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
        ],
        "Rate of Detected Clusters": [
            fem["fpr_anti_reflexive_female"],
            male["fpr_anti_reflexive_female"],
            fem["fpr_anti_reflexive_male"],
            male["fpr_anti_reflexive_male"],
            fem["tpr_reflexive"],
            male["tpr_reflexive"],
        ],
    }

    detailed_data = pd.DataFrame(detailed_data).T

    # Set the first row as the column names
    detailed_data.columns = detailed_data.iloc[0]

    # Drop the first row
    detailed_df = detailed_data.iloc[1:]
    results = [simple_df, detailed_df]

    return results
