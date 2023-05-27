import warnings
warnings.simplefilter("ignore", FutureWarning)


class Evaluator:
    """Module for detecting gender bias in Danish language models."""

    def __init__(self, model_name):
        self.model_name = model_name
        print(
            f"[INFO] You can test {self.model_name} by running Evaluator.evaluate_<model type>()"
        )


    def evaluate_pretrained(
        self, test, mask_token=None, start_token=None, sep_token=None
    ):
        """Evaluate gender bias in a pre-trained model trained with masked language modeling.

        This function can be used for running two different tests:
        The Dawinobias Language Modeling Task and the ABC Language Modeling Task.
        Read more about the specifics of these test in the User Guide.

        Args:
            test (str): choose between "abc" or "dawinobias"
            mask_token (str, optional): mask token of tested model. Specify when running test "abc". Defaults to None.
            start_token (str, optional): start token of tested model. Specify when running test "abc". Defaults to None.
            sep_token (str, optional): sep token of tested model. Specify when running test "dawinobias". Defaults to None.

        Returns:
            list (df): Performance output as list. First element: performance in condensed form. Second element: performance in detailed form.

        *EXAMPLE*

           ```python
           from genda_lens import Evaluator

           # initiate evaluator
           ev = Evaluator(model_name="huggingface-modelname")

           # run abc test
           output = ev.evaluate_pretrained(test="abc", mask_token="<mask>", start_token="<s>", sep_token="</s>")

           # retrieve output
           simple_output = output[0]
           detailed_output = output[1]

           ```
        """
        import pandas as pd
        import spacy
        from transformers import pipeline

        from .lm_tasks.abc_utils import get_output, load_abc, load_mdl, run_abc
        from .lm_tasks.wino_utils import evaluate_lm_winobias, run_winobias

        ### RUN ABC
        # load data
        if test == "abc":
            if start_token is None:
                raise ValueError(
                    "Please specify input argument 'start_token'(str) when running the ABC language modeling task."
                )
            if sep_token is None:
                raise ValueError(
                    "Please specify input argument 'sep_token'(str) when running the ABC language modeling task."
                )
            else:
                pass

            print(f"[INFO] Running the ABC language modeling task on {self.model_name}")
            refl_sents_m, refl_sents_f = load_abc()
            # load tokenizer and model
            model, tokenizer = load_mdl(self.model_name)

            # create results df
            out_df_f = run_abc(
                refl_sents_f, "female", tokenizer, model, start_token, sep_token
            )
            out_df_m = run_abc(
                refl_sents_m, "male", tokenizer, model, start_token, sep_token
            )

            # evaluate abc
            results = get_output(out_df_f, out_df_m, model_name=self.model_name)

        elif test == "dawinobias":
            if mask_token is None:
                raise ValueError(
                    "Please specify input argument 'mask_token'(str) when running the DaWinobias language modeling task."
                )
            else:
                pass
            print(
                f"[INFO] Running the DaWinobias language modeling task on {self.model_name}"
            )
            # load model used for tokenization
            try:
                tokenizer = spacy.load("da_core_news_sm")
            except OSError:
                print("[INFO] Downloading tokenizer: da_core_news_sm from spaCy.")
                from spacy.cli.download import download

                download("da_core_news_sm")
                tokenizer = spacy.load("da_core_news_sm")

            # initiate pipeline
            print(f"[INFO] Loading model {self.model_name} from Hugging Face.")
            nlp = pipeline(task="fill-mask", model=self.model_name)

            # run wino
            clf_rep_anti, clf_rep_pro = run_winobias(
                tokenizer, nlp, mask_token=mask_token, model_name=self.model_name
            )

            results = evaluate_lm_winobias(
                clf_rep_anti, clf_rep_pro, model_name=self.model_name
            )

        else:
            raise ValueError("Not a valid test. Choose between 'abc' and 'dawinobias'")
        print(
            "[INFO] output(list) generated. Access condensed output with output[0], and detailed output with with output[1]."
        )
        return results


    def evaluate_ner(self, n):
        """Evaluate gender bias in a NER model.
        This function can be used for running the DaNe dataset test.
        Read more about the specifics of these test in the User Guide.

        Args:
            n (int): Number of repetitions to run the augmentation pipeline. To ensure robustness we recommend a value of n => 20.

        Returns:
            list (df): Performance output as list. First element: performance in condensed form. Second element: performance in detailed form.

        *EXAMPLE*

           ```python
            from genda_lens import Evaluator

            # initiate evaluator
            ev = Evaluator(model_name="huggingface-modelname")

            # run test
            output = ev.evaluate_ner(n=20)

            # retrieve output
            simple_output = output[0]
            detailed_output = output[1]

           ```
        """
        from dacy.datasets import dane

        from .ner_tasks.augmentation import f_aug, m_aug, muslim_f_aug, muslim_m_aug
        from .ner_tasks.performance import load_mdl, eval_model_augmentation

        testdata = dane(
            splits=["test"], redownload=True, open_unverified_connected=True
        )

        model = load_mdl(self.model_name)
        if n <= 1:
            print(
                f"[INFO] Please choose a value for n larger than 1 to ensure robustness, got: {n}."
            )
            print(
                f"[INFO] Running the NER task on {self.model_name} with low value for n."
            )
        else:
            print(f"[INFO] Running the NER task on {self.model_name}")

        # define augmenters
        augmenters = [
            (f_aug, "Majority female names", n),
            (m_aug, "Majority male names", n),
            (muslim_f_aug, "Minority female names", n),
            (muslim_m_aug, "Minority male names", n),
        ]

        # run model
        output = eval_model_augmentation(
            model, self.model_name, str(n), augmenters, testdata
        )
        print(
            "[INFO] output(list) generated. Access condensed output with output[0], and detailed output with with output[1]."
        )
        return output

    def evaluate_coref(self, test, model):
        """Evaluate gender bias in a coreference model.

        This function can be used for running two different tests:
        The Dawinobias Language Coreference Task and the ABC Coreference Task.
        Read more about the specifics of these test in the User Guide.

        Args:
            test (str): choose between "abc" or "dawinobias"
            model (_type_): a coreference model object.

        Returns:
            list (df): Performance output as list. First element: performance in condensed form. Second element: performance in detailed form.

        *EXAMPLE*

           ```python
           from genda_lens import Evaluator

           # load coref model
           from danlp import load_xlmr_coref_model
           model = load_xlmr_coref_model()

           # initiate evaluator
           ev = Evaluator(model_name="danlp-xlmr")

           # run abc test
           output = ev.evaluate_coref(test="abc", model=model)

           # retrieve output
           simple_output = output[0]
           detailed_output = output[1]
           ```
        """
        import os
        import random
        import sys

        # import json
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import spacy
        import torch
        from sklearn.metrics import classification_report, f1_score

        if test == "dawinobias":
            import nltk
            from .coref_tasks.wino_utils import (
                evaluate_coref_winobias,
                run_winobias_coref,
            )

            nltk.download("omw-1.4")  # only wino
            # load model used for tokenization
            try:
                nlp = spacy.load("da_core_news_sm")
            except OSError:
                print("[INFO] Downloading tokenizer: da_core_news_sm from spaCy.")
                from spacy.cli.download import download

                download("da_core_news_sm")
                nlp = spacy.load("da_core_news_sm")

            print(
                f"[INFO] Running the DaWinobias coreference task on {self.model_name}"
            )
            anti_res, pro_res = run_winobias_coref(model, nlp)
            results = evaluate_coref_winobias(
                anti_res, pro_res, model_name=self.model_name
            )

        elif test == "abc":
            import pandas as pd
            from .coref_tasks.abc_utils import (
                eval_results,
                evaluate_coref_abc,
                run_abc_coref,
            )

            print(f"[INFO] Running the ABC coreference task on {self.model_name}")
            fem_preds, male_preds = run_abc_coref(model)
            # two dicts of f1 scores
            df_fem, df_male, all_sents = evaluate_coref_abc(
                fem_preds=fem_preds, male_preds=male_preds
            )

            results = eval_results(
                df_fem, df_male, all_sents, model_name=self.model_name
            )
        else:
            raise ValueError("Not a valid test. Choose between 'abc' and 'dawinobias'")
        print(
            "[INFO] output(list) generated. Access condensed output with output[0], and detailed output with with output[1]."
        )
        return results


class Visualizer:
    """Module for visualizing results from the bias evaluation."""

    def __init__(self):
        import seaborn as sns

        sns.set_style("whitegrid", rc={"lines.linewidth": 1})
        sns.set_context("notebook", font_scale=1.2)

    def visualize_results(self, data, framework, model_name, task=None):
        """Visualize output from any of the genderbias tests that can be run in this package.

        Args:
            data (df): detailed output from any of the tests.
            framework (str): choose between "ner", "dawinobias" or "abc".
            model_name (str): model name
            task (str, optional): choose between "lm", "ner" or "coref" depending on which task the output is from.
        Returns:
            plot (plot): seaborn plot visualization.

        *EXAMPLE*

           ```python
           from genda_lens import Visualizer

           # initiate visualizer
           viz = Visualizer()

           # visualize ner results
           plot = viz.visualize_results(data = detailed_output_ner, framework = "ner", model_name "my-model-name")

           # visualize abc lm results
           plot = viz.visualize_results(data = detailed_output_lm, framework = "abc", model_name "my-model-name", task="lm")

           # visualize abc coref results
           plot = viz.visualize_results(data = detailed_output_lm, framework = "abc", model_name "my-model-name", task="coref")
           ```
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sns.set_style("whitegrid", rc={"lines.linewidth": 10})

        df = pd.DataFrame()

        if framework == "abc":
            df["Stereotypical Occupations"] = [
                "Female occupations",
                "Male occupations",
                "Main Effect",
            ] * 2
            df["Anti-reflexive Pronoun"] = ["Female"] * 3 + ["Male"] * 3
            markers = ["o", "o", "o"]
            x = "Anti-reflexive Pronoun"
            nuance = "Stereotypical Occupations"

            if task == "coref":
                try:
                    mean_fem = list(data.loc["Mean Rate of Detected Clusters"])[1]
                    mean_male = list(data.loc["Mean Rate of Detected Clusters"])[3]
                    fpr_fem_pron_fem_occ = list(data.loc["Rate of Detected Clusters"])[
                        0
                    ]
                    fpr_fem_pron_male_occ = list(data.loc["Rate of Detected Clusters"])[
                        1
                    ]
                    fpr_male_pron_fem_occ = list(data.loc["Rate of Detected Clusters"])[
                        2
                    ]
                    fpr_male_pron_male_occ = list(
                        data.loc["Rate of Detected Clusters"]
                    )[3]
                except:
                    mean_fem = float(data.iloc[2, 2])
                    mean_male = float(data.iloc[2, 4])
                    fpr_fem_pron_fem_occ = float(data.iloc[4, 1])
                    fpr_fem_pron_male_occ = float(data.iloc[4, 2])
                    fpr_male_pron_fem_occ = float(data.iloc[4, 3])
                    fpr_male_pron_male_occ = float(data.iloc[4, 4])

                points = [
                    fpr_fem_pron_fem_occ,
                    fpr_fem_pron_male_occ,
                    mean_fem,
                    fpr_male_pron_fem_occ,
                    fpr_male_pron_male_occ,
                    mean_male,
                ]

                df["False Positive Rates"] = points
                y = "False Positive Rates"
                title = f"ABC Coref Task: {model_name}"

            elif task == "lm":
                try:
                    points = [
                        float(data.iloc[4, 1].split(" ")[0]),
                        float(data.iloc[4, 2].split(" ")[0]),
                        float(data.iloc[2, 1].split(" ")[0]),
                        float(data.iloc[4, 3].split(" ")[0]),
                        float(data.iloc[4, 4].split(" ")[0]),
                        float(data.iloc[2, 3].split(" ")[0]),
                    ]
                except:
                    points = [
                        float(df.iloc[4, 0].split(" ")[0]),
                        float(data.iloc[4, 1].split(" ")[0]),
                        float(data.iloc[2, 0].split(" ")[0]),
                        float(data.iloc[4, 2].split(" ")[0]),
                        float(data.iloc[4, 3].split(" ")[0]),
                        float(data.iloc[2, 2].split(" ")[0]),
                    ]
                df["Median Perplexity"] = points
                y = "Median Perplexity"
                title = f"ABC LM Task: {model_name}"

        elif framework == "dawinobias":
            df["Pronoun"] = ["Female (F1)", "Male (F1)", "Main Effect (Accuracy)"] * 2
            if task == "coref":
                try:
                    points = [
                        float(data.iloc[4, 1]),
                        float(data.iloc[4, 2]),
                        float(data.iloc[2, 1]),
                        float(data.iloc[4, 3]),
                        float(data.iloc[4, 4]),
                        float(data.iloc[2, 3]),
                    ]
                except:
                    points = [
                        data.loc["F1"][0],
                        data.loc["F1"][1],
                        data.loc["Accuracy"][0],
                        data.loc["F1"][2],
                        data.loc["F1"][3],
                        data.loc["Accuracy"][2],
                    ]
                title = f"DaWinoBias, Coreference Task: {model_name}"
            elif task == "lm":
                try:  # if loaded
                    points = [
                        float(data.iloc[4, 1]),
                        float(data.iloc[4, 2]),
                        float(data.iloc[2, 1]),
                        float(data.iloc[4, 3]),
                        float(data.iloc[4, 4]),
                        float(data.iloc[2, 3]),
                    ]
                except:  # if
                    points = [
                        data.loc["F1"][0],
                        data.loc["F1"][1],
                        data.loc["Accuracy"][0],
                        data.loc["F1"][2],
                        data.loc["F1"][3],
                        data.loc["Accuracy"][2],
                    ]
                title = f"DaWinoBias, LM Task: {model_name}"

            df["Performance"] = points
            df["Condition"] = ["Anti-stereotypical"] * 3 + ["Pro-stereotypical"] * 3
            x = "Condition"
            y = "Performance"
            nuance = "Pronoun"
            markers = ["o", "o", "o"]
            title = title

        elif framework == "ner":
            df["Protected Group"] = ["Majority (F1)", "Minority (F1)"] * 2
            try:
                points = [
                    float(data.iloc[2, 1].split(" ")[0]),
                    float(data.iloc[4, 2].split(" ")[0]),
                    float(data.iloc[2, 3].split(" ")[0]),
                    float(data.iloc[4, 4].split(" ")[0]),
                ]
            except:
                points = [
                    float(data.iloc[4, 0].split(" ")[0]),
                    float(data.iloc[4, 1].split(" ")[0]),
                    float(data.iloc[4, 2].split(" ")[0]),
                    float(data.iloc[4, 3].split(" ")[0]),
                ]

            df["Performance"] = points
            df["Augmentation"] = ["Female Names"] * 2 + ["Male Names"] * 2
            x = "Augmentation"
            y = "Performance"
            nuance = "Protected Group"
            markers = ["o", "o"]
            title = f"NER Task, Augmented DaNe: {model_name}"

        sns.pointplot(
            data=df,
            x=x,
            y=y,
            hue=nuance,
            dodge=True,
            join=True,
            markers=markers,
            scale=1.2,
            linestyles=[":", ":", "-"],
            palette=["sandybrown", "mediumpurple", "darkgrey"],
        ).set_title(title)

        plt.minorticks_on()

        return plt
