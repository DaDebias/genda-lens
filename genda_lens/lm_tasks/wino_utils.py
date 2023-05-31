"""
Utility functions for running the DaWinobias language modeling task.

Builds on the codebase developed for the following project: 

Title: “DaWinoBias: Assessing Occupational Gender Stereotypes in Danish NLP Models”
Authors: Koppelgaard, K., Brødbæk, S. K.
Date: 2021
Code availability: https://github.com/NLP-exam/DaWinoBias
"""
import math
import os
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm


def load_texts(filepath):
    """load DaWinoBias texts - and shuffle data"""
    with open(filepath) as file:
        text = file.read().splitlines()
    lines = [line.strip() for line in text]
    random.shuffle(lines)
    return lines


def remove_sq_br(tokens):
    # input tokens to remove '[]'
    return [[token for token in tokens if token != "[" and token != "]"]]


def idx_pron(tokens):
    """Get index of pronoun in sentence"""
    # define pronouns
    pronouns = [
        "hans",
        "hendes",
        "han",
        "hun",
        "ham",
        "hende",
        "▁hun",
        "▁hendes",
        "▁hende",
        "▁han",
        "▁hans",
        "▁ham",
    ]
    # find idx of pronouns
    prons_idx = [tokens.index(i) for i in pronouns if i in tokens][0]
    return prons_idx


def predict_masked(lines, condition, nlp, tokenizer, mask_token, model_name):
    labels, preds = [], []

    print(f"[INFO] Running test on condition: {condition} sentences.")
    for idx, line in enumerate(tqdm(lines)):
        # tokenize and lowercase
        line_tokenized = tokenizer(line)
        tokens = [token.text.lower() for token in line_tokenized]

        # remove square brackets from tokens
        tokens = remove_sq_br(tokens)[0]

        # find index of pronoun
        prons_idx = idx_pron(tokens)

        # save correct pronoun
        correct_pronoun = tokens[prons_idx]

        # MASK pronouns
        tokens[prons_idx] = mask_token

        # join sentence into one string
        if mask_token == "<mask>" and model_name == "vesteinn/DanskBert":
            sentence = " ".join(tokens).replace(" <mask>", "<mask>")
        else:
            sentence = " ".join(tokens)

        # compute fill-mask prediction
        pred = nlp(sentence)[0]["token_str"]

        # save labels and predictions
        labels.append(correct_pronoun)
        preds.append(pred)

    return labels, preds


def group_pronouns(pronouns):
    """group labels in female and male pronouns"""

    # define groups of pronouns
    female_pronouns = ["hun", "hendes", "hende", "▁hun", "▁hendes", "▁hende"]
    male_pronouns = ["han", "hans", "ham", "▁han", "▁hans", "▁ham"]

    # initialize lists
    group_pronoun = []

    # group labels
    for pronoun in pronouns:
        if pronoun in female_pronouns:
            group_pronoun.append("hun/hendes")
        elif pronoun in male_pronouns:
            group_pronoun.append("han/hans")
        else:
            group_pronoun.append("UNK")
    return group_pronoun


def evaluate_model(labels, predictions):  # , filename, model_name):
    """Create and save classification report
    Args:
        labels: labels
        predictions: model predictions
    Returns:
        clf_report: classification report in pandas df format
    """
    # create df for storing metrics
    df = pd.DataFrame(
        classification_report(labels, predictions, output_dict=True, zero_division=1)
    ).round(decimals=2)
    return df


def run_winobias(tokenizer, nlp, mask_token, model_name):
    # data paths
    inpath_pro = os.path.join(
        Path(__file__).parents[1], "data", "DaWinoBias_pro_stereotyped_evalda.txt"
    )
    inpath_anti = os.path.join(
        Path(__file__).parents[1], "data", "DaWinoBias_anti_stereotyped_evalda.txt"
    )

    # load data
    anti_sents = load_texts(inpath_anti)
    pro_sents = load_texts(inpath_pro)

    # mask and predict pronoun
    anti_labels_, anti_preds_ = predict_masked(
        lines=anti_sents,
        condition="anti-stereotypical",
        nlp=nlp,
        tokenizer=tokenizer,
        mask_token=mask_token,
        model_name=model_name,
    )
    pro_labels_, pro_preds_ = predict_masked(
        lines=pro_sents,
        condition="pro-stereotypical",
        nlp=nlp,
        tokenizer=tokenizer,
        mask_token=mask_token,
        model_name=model_name,
    )

    # group pronouns into male/female category
    anti_labels, anti_preds = group_pronouns(anti_labels_), group_pronouns(anti_preds_)
    pro_labels, pro_preds = group_pronouns(pro_labels_), group_pronouns(pro_preds_)

    # print( "[INFO] Raw predictions for model:", model_name)
    # print("[INFO] Number of female pron. predictions:", len([i for i in anti_preds if i =='hun/hendes']))
    # print("[INFO] Number of male pron. predictions:", len([i for i in anti_preds if i =='han/hans']))
    # print("[INFO] Number of 'UNKNOWN' predictions:", len([i for i in anti_preds if i =='UNK']))

    # evaluate performance
    clf_rep_anti = evaluate_model(anti_labels, anti_preds)
    clf_rep_pro = evaluate_model(pro_labels, pro_preds)

    return clf_rep_anti, clf_rep_pro


def logratio(x: float, y: float):
    return round(math.log2(x / y), 2)


def evaluate_lm_winobias(anti_res, pro_res, model_name):
    """Evaluate winobias coref experiment
    Args:
        anti_res: results for anti-stereotypical condition
        pro_res: results for pro-stereotypical condition

    """
    # Gender Effect Size calculation
    anti_acc = anti_res["accuracy"].values[0]
    pro_acc = pro_res["accuracy"].values[0]
    gender_effect_size = logratio(pro_acc, anti_acc)

    anti_f1_fem_pron = anti_res["hun/hendes"].values[2]
    anti_f1_male_pron = anti_res["han/hans"].values[2]

    pro_f1_fem_pron = pro_res["hun/hendes"].values[2]
    pro_f1_male_pron = pro_res["han/hans"].values[2]

    simlpe_data = {
        f"Simple Output for {model_name}": ["", "DaWinoBias"],
        "Gender Effect Size": ["", gender_effect_size],
        "Condition": ["Anti-stereotypical", "Pro-stereotypical"],
        "Accuracy": [anti_acc, pro_acc],
    }

    simlpe_data = pd.DataFrame(simlpe_data).T

    # Set the first row as the column names
    simlpe_data.columns = simlpe_data.iloc[0]

    # Drop the first row
    simple_df = simlpe_data.iloc[1:]

    detailed_data = {
        f"Detailed Output for {model_name}": ["", "DaWinoBias", "", ""],
        "Gender Effect Size": ["", gender_effect_size, "", ""],
        "Condition": ["Anti-stereotypical", "", "Pro-stereotypical", ""],
        "Accuracy": [anti_acc, "", pro_acc, ""],
        "Pronouns": ["Female", "Male", "Female", "Male"],
        "F1": [anti_f1_fem_pron, anti_f1_male_pron, pro_f1_fem_pron, pro_f1_male_pron],
    }

    detailed_data = pd.DataFrame(detailed_data).T

    # Set the first row as the column names
    detailed_data.columns = detailed_data.iloc[0]

    # Drop the first row
    detailed_df = detailed_data.iloc[1:]

    results = [simple_df, detailed_df]

    return results
