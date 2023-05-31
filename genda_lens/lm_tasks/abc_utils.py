"""
Utility functions for running the ABC language modeling task.

Builds on the codebase developed for the following paper: 

Title: “Type B reflexivization as an unambiguous testbed for multilingual multi-task gender bias.”
Authors: González, A. V., Barrett, M., Hvingelby, R., Webster, K., & Søgaard, A.
Date: 2020
Code availability: https://github.com/anavaleriagonzalez/ABC-dataset
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_mdl(model_name):
    print(f"[INFO] Loading model {model_name} from Hugging Face.")
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def load_abc():
    from pathlib import Path

    inpath_male = os.path.join(Path(__file__).parents[1], "data", "abc_male_sents.txt")
    inpath_fem = os.path.join(Path(__file__).parents[1], "data", "abc_fem_sents.txt")
    male_sents = load_sents(inpath_male)
    fem_sents = load_sents(inpath_fem)
    return male_sents, fem_sents


def load_sents(filename):
    """
    A function that loads all the reflexive sentences from the ABC dataset.
    """
    reflexive_sents = []
    with open(filename, "r") as f:
        lines = f.readlines()

        restart = 0
        for line in lines:
            if "--------------" in line:
                pass
            elif "---" in line:
                restart = 0
            else:
                if restart == 0:
                    reflexive_sents.append(line.strip())
                    restart = 1
    return reflexive_sents


def tokenize_sentence(sentence, tokenizer, start_token, sep_token):
    """
    A function that tokenizes the reflexive sentences.
    """
    sentence = start_token + " " + sentence + " " + sep_token
    tokenize_input = tokenizer.tokenize(sentence)
    return tokenize_input


def get_pron_index(sent, pronoun_list):
    """
    A function that takes the tokenized reflexive sent and
    locates index of pronoun
    replaces pronoun with male/female pronoun
    returns 2 augmented sentences
    """
    # get index of pronoun
    no_pron = True
    for i, token in enumerate(sent):
        if token in pronoun_list:
            pron_index = i
            no_pron = False
            break
        else:
            pass
    if no_pron == True:
        return "no pronouns to replace"
    else:
        return pron_index


def get_augmented_sents(tokenize_input, idx):
    """
    Take original sentence and replace pronoun with antirreflexive pronouns (hans/hendes)
    """
    tokenize_mask_male = tokenize_input.copy()
    tokenize_mask_female = tokenize_input.copy()
    tokenize_mask_male[int(idx)] = "hans"
    tokenize_mask_female[int(idx)] = "hendes"
    return tokenize_mask_male, tokenize_mask_female


def create_tensors(truth, male, fem, tokenizer):
    """
    Make augmented sentences into tensors.
    """
    tensor_truth = torch.tensor([tokenizer.convert_tokens_to_ids(truth)])
    tensor_male = torch.tensor([tokenizer.convert_tokens_to_ids(male)])
    tensor_fem = torch.tensor([tokenizer.convert_tokens_to_ids(fem)])
    return tensor_truth, tensor_male, tensor_fem


def get_predictions(tensor_truth, tensor_male, tensor_fem, model):
    """
    A function that takes the 3 sentences and returns model predictions for each of them.
    Used to compute loss.
    """
    with torch.no_grad():
        pred_truth = model(tensor_truth)[0]
    with torch.no_grad():
        pred_male = model(tensor_male)[0]
    with torch.no_grad():
        pred_fem = model(tensor_fem)[0]
    return pred_truth, pred_male, pred_fem


def compute_loss(loss_fct, pred_truth, pred_male, pred_fem, tensor_truth):
    """
    Take model predictions and compute loss for all 3 sentences
    by comparing to pred_truth, which is the prediction of the sentence with the correct (reflexive) pronoun.
    """
    loss_male = loss_fct(pred_male.squeeze(), tensor_truth.squeeze()).data
    loss_fem = loss_fct(pred_fem.squeeze(), tensor_truth.squeeze()).data
    loss_ref = loss_fct(pred_truth.squeeze(), tensor_truth.squeeze()).data
    loss_list = [loss_male, loss_fem, loss_ref]
    return loss_list


def score_sent(sent, loss_fct, tokenizer, model, pron_list, start_token, sep_token):
    """
    Tak a sentence with a relflexive pronoun,
    replace pronoun with antireflexives (male/female pronoun)
    and compute loss and perplexity for all 3 sentences.
    Args:
        sent (str): A sentence with reflexive pronouns.
        loss_fct (torch.nn.CrossEntropyLoss): A loss function.
        tokenizer (BertTokenizer): A tokenizer.
        model (BertForMaskedLM): A language model.
        pron_list (list): A list of pronouns.
    Returns:
        Loss and perplexity values for sentence with reflexive, male and female pronoun.
    """
    tokenized_refl = tokenize_sentence(sent, tokenizer, start_token, sep_token)
    index = get_pron_index(tokenized_refl, pron_list)
    tokenized_male, tokenized_fem = get_augmented_sents(tokenized_refl, index)
    tensor_truth, tensor_male, tensor_fem = create_tensors(
        tokenized_refl, tokenized_male, tokenized_fem, tokenizer
    )
    pred_truth, pred_male, pred_fem = get_predictions(
        tensor_truth, tensor_male, tensor_fem, model
    )
    loss_values = compute_loss(loss_fct, pred_truth, pred_male, pred_fem, tensor_truth)
    return (
        str(math.exp(loss_values[0])),
        str(math.exp(loss_values[1])),
        str(math.exp(loss_values[2])),
    )


def run_abc(reflexive_sents, condition, tokenizer, model, start_token, sep_token):
    """
    Run the ABC dataset and write results to file.
    Args:
        outpath (str): Path to output file.
        reflexive_sents (list): A list of sentences with reflexive pronouns.
        loss_fct (torch.nn.CrossEntropyLoss): A loss function.
        tokenizer (BertTokenizer): A tokenizer.
        model (BertForMaskedLM): A language model.
        pron_list (list): A list of pronouns.
    """

    # loss function for making model predictions and loss - used for getting pperplexity scores
    loss_fct = torch.nn.CrossEntropyLoss()

    pronouns_list = ["sin", "sit", "sine", "▁sin", "▁sit", "▁sine"]

    print(f"[INFO] Running test on condition: {condition} occupations.")
    # loop over sentences to compute loss and perplexity
    sents, perp_m, perp_f, perp_r = [], [], [], []
    for idx, sent in enumerate(tqdm(reflexive_sents)):
        scores = score_sent(
            sent, loss_fct, tokenizer, model, pronouns_list, start_token, sep_token
        )
        sents.append(sent)
        perp_m.append(scores[0])
        perp_f.append(scores[1])
        perp_r.append(scores[2])

    df = pd.DataFrame(
        {
            "sent": sents,
            "perplexity_male": perp_m,
            "perplexity_female": perp_f,
            "perplexity_reflexive": perp_r,
        }
    )

    df["perplexity_male"] = df["perplexity_male"].astype(float)
    df["perplexity_female"] = df["perplexity_female"].astype(float)
    df["perplexity_reflexive"] = df["perplexity_reflexive"].astype(float)

    return df


### EVALUTATION
def logratio(x: float, y: float):
    return round(math.log2(x / y), 2)


def calc_median_iqr(df, l_ratio=False):
    # return median value for male/female pronoun with IQR
    # fem
    median_fem = str(df.relative_female.median().round(2))
    iqr_fem = str(tuple(np.percentile(df.relative_female, [25, 75]).round(2)))

    # male
    median_male = str(df.relative_male.median().round(2))
    iqr_male = str(tuple(np.percentile(df.relative_male, [25, 75]).round(2)))
    if l_ratio == True:
        ratio = logratio(
            float(median_fem), float(median_male)
        )  # female over male here! because high perplexity = bad
        return str(median_fem + " " + iqr_fem), str(median_male + " " + iqr_male), ratio
    elif l_ratio == False:
        return str(median_fem + " " + iqr_fem), str(median_male + " " + iqr_male)


def get_condensed(all_occ_FEM_PRON, all_occ_MALE_PRON, lratio, model_name):
    data = {
        f"Simple Output: {model_name}": ["", "ABC", "", ""],
        "Gender Effect Size": ["", lratio, "", ""],
        "Pronoun": ["Female", "", "", "Male"],
        "Perplexity Median (IQR) Relative scores": [
            "",
            all_occ_FEM_PRON,
            "",
            all_occ_MALE_PRON,
        ],
    }
    df = pd.DataFrame(data).T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df


def get_detailed(
    all_occ_FEM_PRON,
    all_occ_MALE_PRON,
    lratio,
    fem_occ_FEM_PRON,
    fem_occ_MALE_PRON,
    male_occ_FEM_PRON,
    male_occ_MALE_PRON,
    model_name,
):
    data = {
        f"Detailed Output for: {model_name}": ["", "ABC", "", ""],
        "Gender Effect Size": ["", lratio, "", ""],
        "Pronoun": ["Female", "", "Male", ""],
        "Perplexity Median (IQR) Relative scores": [
            all_occ_FEM_PRON,
            "",
            all_occ_MALE_PRON,
            "",
        ],
        "Stereotypical Occupation": ["Female", "Male", "Female", "Male"],
        "Perplexity Median (IQR) Relative scores ": [
            fem_occ_FEM_PRON,
            male_occ_FEM_PRON,
            fem_occ_MALE_PRON,
            male_occ_MALE_PRON,
        ],
    }
    df = pd.DataFrame(data).T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df


def get_output(df_FEM_occ, df_MALE_occ, model_name):
    # calculate relative perplexities
    df_FEM_occ["relative_female"] = (
        df_FEM_occ["perplexity_female"] - df_FEM_occ["perplexity_reflexive"]
    )
    df_FEM_occ["relative_male"] = (
        df_FEM_occ["perplexity_male"] - df_FEM_occ["perplexity_reflexive"]
    )
    df_MALE_occ["relative_female"] = (
        df_MALE_occ["perplexity_female"] - df_MALE_occ["perplexity_reflexive"]
    )
    df_MALE_occ["relative_male"] = (
        df_MALE_occ["perplexity_male"] - df_MALE_occ["perplexity_reflexive"]
    )

    # create combined df
    ALL_occ = pd.concat([df_FEM_occ, df_MALE_occ], ignore_index=True)

    # calculate median and IQR's
    fem_occ_FEM_PRON, fem_occ_MALE_PRON = calc_median_iqr(df_FEM_occ)
    male_occ_FEM_PRON, male_occ_MALE_PRON = calc_median_iqr(df_MALE_occ)
    all_occ_FEM_PRON, all_occ_MALE_PRON, log_ratio = calc_median_iqr(
        ALL_occ, l_ratio=True
    )

    # get condensed output
    out_condensed = get_condensed(
        all_occ_FEM_PRON, all_occ_MALE_PRON, log_ratio, model_name
    )

    # get detailed output
    out_detailed = get_detailed(
        all_occ_FEM_PRON,
        all_occ_MALE_PRON,
        log_ratio,
        fem_occ_FEM_PRON,
        fem_occ_MALE_PRON,
        male_occ_FEM_PRON,
        male_occ_MALE_PRON,
        model_name,
    )

    return [out_condensed, out_detailed]
