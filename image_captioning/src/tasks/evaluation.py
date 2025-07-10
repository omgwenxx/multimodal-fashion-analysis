import argparse
import sys

sys.path.append("..")
from pyevalcap.scorer import Scorer
from pyevalcap.accuracy.accuracy import Accuracy, compute_accuracy
from pyevalcap.map.map import Map
from preprocessing.preprocess_hm import add_attrs_column_batch, add_attrs_column
import os
import pandas as pd
import json
import torch

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
CONTROL_FILES_DIR = os.getenv("CONTROL_FILES_DIR")
CAPTIONS_DIR = os.path.join(RESULTS_DIR, "caption_results")
ATTRS_DIR = os.path.join(RESULTS_DIR, "attrs_results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics_results")
CLASS_DIR = os.path.join(RESULTS_DIR, "class_results")

models = [
    "blip2-opt-2.7b",
    "blip2-opt-6.7b",
    "blip2-flan-t5-xl",
    "blip2-flan-t5-xxl",
    "llava-1.5-7b-hf",
    "llava-1.5-13b-hf",
]
datasets = ["hm", "fic"]


def round_value(value):
    return round(round(value, 3) * 100, 1)


def load_json(file_path: str):
    """
    Loads a JSON file.

    Parameters:
        file_path (str): The file path.

    Returns:
        dict: The loaded JSON file.
    """
    with open(file_path, "r") as j:
        file = json.load(j)
    return file


def compute_score(model, dataset: str, finetuned: bool = False) -> pd.DataFrame:
    """
    Computes the scores for a single model and dataset.

    Parameters:
        model (str): The model name.
        dataset (str): The dataset name.
        finetuned (bool): Whether the model is finetuned or not.

    Returns:
        pd.DataFrame: A DataFrame containing the computed scores.
    """
    results = pd.DataFrame()
    finetuned_postfix = "_finetuned" if finetuned else ""
    model_name = f"{dataset}_{model}{finetuned_postfix}"
    result_prefix = os.path.join(CAPTIONS_DIR, dataset, model_name)
    gts_file_path = f"{result_prefix}_gts.json"
    res_file_path = f"{result_prefix}_res.json"

    try:
        GT = load_json(gts_file_path)
        RES = load_json(res_file_path)

        IDs = GT.keys()
        scorer = Scorer()
        metrics = scorer.evaluate(GT, RES, IDs)
        metrics["model"] = model_name
        metrics["dataset"] = dataset

        results = pd.DataFrame([metrics])
    except FileNotFoundError:
        print(f"Error: File not found for {model}{finetuned_postfix} and {dataset} at {res_file_path}")

    return results


def score_mode(finetuned: bool = False) -> pd.DataFrame:
    """
    Runs through all models and datasets to compute scores.

    Parameters:
        finetuned (bool): Whether the models are finetuned or not.

    Returns:
        pd.DataFrame: A DataFrame containing all computed scores.
    """
    all_results = pd.DataFrame()

    finetuned_postfix = "_finetuned" if finetuned else ""
    for model in models:
        for dataset in datasets:
            print(f"\n{b_}Computing score for {dataset} dataset and model {model}{finetuned_postfix}{sr_}")
            results = compute_score(model, dataset, finetuned)
            all_results = pd.concat([all_results, results], ignore_index=True)

    # Sort and reorder columns
    sorted_df = all_results.sort_values(by=["dataset"], ascending=False)
    columns = ["dataset", "model"] + [col for col in sorted_df.columns if col not in ["dataset", "model"]]
    sorted_df = sorted_df[columns]

    os.makedirs(METRICS_DIR, exist_ok=True)

    # Save raw results
    sorted_df.to_csv(
        os.path.join(METRICS_DIR, f"results_score_raw{finetuned_postfix}.csv"),
        index=False,
    )

    # Round specified metrics
    metrics = [
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
        "CIDEr",
        "SPICE",
    ]
    sorted_df[metrics] = sorted_df[metrics].apply(round_value)

    # Save rounded results
    sorted_df.to_csv(os.path.join(METRICS_DIR, f"results_score{finetuned_postfix}.csv"), index=False)

    return sorted_df


def compute_category_accuracy(model, dataset:str, finetuned: bool = False, full_run: bool = True) -> pd.DataFrame:

    finetuned_postfix = "_finetuned" if finetuned else ""
    print(f"\n{b_}Working on {dataset} dataset and model {model}{finetuned_postfix}{sr_}")
    print(f"Full run") if full_run else print(f"Using precomputed class predictions")

    cates_file = os.getenv("HM_CATES") if dataset == "hm" else os.getenv("FIC_CATES")
    gts = load_json(cates_file)

    results = pd.DataFrame(columns=["dataset", "model", "accuracy"])

    acc = None
    if full_run:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        acc = Accuracy(dataset, device)

    try:
        result_prefix = os.path.join(CAPTIONS_DIR, dataset, f"{dataset}_{model}{finetuned_postfix}")
        res_file_path = f"{result_prefix}_res.json"

        RES = load_json(res_file_path)

        filename = f"{dataset}_{model}{finetuned_postfix}_class.csv"
        file_path = os.path.join(os.path.join(CLASS_DIR, dataset), filename)

        if full_run:
            preds = acc.compute_class(RES)

            os.makedirs(os.path.join(CLASS_DIR, dataset), exist_ok=True)
            
            print("Saving predictions to", file_path)
            preds_df = pd.DataFrame({"preds": preds, "gts": gts})
            preds_df.to_csv(file_path, index=False)
        else:
            preds = pd.read_csv(file_path)

        accuracy = compute_accuracy(preds, gts)

        new_row = {"dataset": dataset, "model": model, "accuracy": accuracy}
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        print(f"Error: File not found for {model}{finetuned_postfix} and {dataset} at {res_file_path}")

    # free up GPU space
    del acc

    return results


def accuracy_mode(finetuned: bool = False, full_run: bool = True) -> pd.DataFrame:
    finetuned_postfix = "_finetuned" if finetuned else ""

    all_results = pd.DataFrame(columns=["dataset", "model", "accuracy"])

    for model in models:
        for dataset in datasets:
            results = compute_category_accuracy(model, dataset, finetuned=finetuned, full_run=full_run)
            all_results = pd.concat([all_results, results], ignore_index=True)

    os.makedirs(METRICS_DIR, exist_ok=True)

    sorted_df = all_results.sort_values(by=["dataset"], ascending=False)
    sorted_df.to_csv(os.path.join(METRICS_DIR, f"results_acc_raw{finetuned_postfix}.csv"),index=False)
    print("Saved file to",os.path.join(METRICS_DIR, f"results_acc_raw{finetuned_postfix}.csv"))

    sorted_df["accuracy"] = sorted_df["accuracy"].round(3)
    sorted_df.to_csv(os.path.join(METRICS_DIR, f"results_acc{finetuned_postfix}.csv"), index=False)

    return sorted_df


def compute_map(model, dataset, finetuned: bool = False) -> pd.DataFrame:
    finetuned_postfix = "_finetuned" if finetuned else ""
    print(f"\n{b_}Working with dataset {dataset} and model {model}{finetuned_postfix}{sr_}")

    gts_path = os.getenv("HM_TEST_ATTRS") if dataset == "hm" else os.getenv("FIC_ATTRS")
    attrs_path = os.getenv("HM_TRAIN_ATTRS") if dataset == "hm" else os.getenv("FIC_ATTRS_ALL")   

    print(f"Using attributes for GTS from {gts_path} and pool of attrs from {attrs_path}")
    
    gts = load_json(gts_path)
    attrs = load_json(attrs_path)

    results = pd.DataFrame(columns=["dataset", "model", "mAP", "mAR"])

    try:
        result_prefix = os.path.join(ATTRS_DIR, dataset, f"{dataset}_{model}{finetuned_postfix}")
        res_file_path = f"{result_prefix}_attrs.json"

        RES = load_json(res_file_path)

        # call func(gts, preds)
        map = Map()
        data = map.compute_score(gts, RES, attrs)

        # print for sanity check of number of results vs. number of items
        print(f"Number of items: {len(RES)}")
        print(f"Number of results: {len(data)}, should be number of items - 1")
        
        mean_precision = round_value(sum(data["precision"]) / len(data["precision"]))
        mean_recall = round_value(sum(data["recall"]) / len(data["recall"]))

        new_row = {
            "dataset": dataset,
            "model": model,
            "mAP": mean_precision,
            "mAR": mean_recall,
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        # save control files
        os.makedirs(os.path.join(CONTROL_FILES_DIR, dataset), exist_ok=True)
        filename = f"{dataset}_{model}{finetuned_postfix}_control.csv"
        file_path = os.path.join(os.path.join(CONTROL_FILES_DIR, dataset), filename)
        data.to_csv(file_path, index=False)

        print(f"Mean Precision: {mean_precision}")
        print(f"Mean Recall: {mean_recall}")
    except FileNotFoundError:
        print(f"Error: File not found for {model}{finetuned_postfix} and {dataset} at {res_file_path}")

    return results


def map_mode(finetuned: bool = False) -> pd.DataFrame:
    all_results = pd.DataFrame(columns=["dataset", "model", "mAP", "mAR"])

    for model in models:
        for dataset in datasets:
            results = compute_map(model, dataset, finetuned)
            all_results = pd.concat([all_results, results], ignore_index=True)

    os.makedirs(METRICS_DIR, exist_ok=True)

    finetuned_postfix = "_finetuned" if finetuned else ""
    sorted_df = all_results.sort_values(by=["dataset", "model"], ascending=False)
    sorted_df.to_csv(os.path.join(METRICS_DIR, f"results_map{finetuned_postfix}.csv"), index=False)
    print("Saved file to",os.path.join(METRICS_DIR, f"results_map{finetuned_postfix}.csv"))

    return sorted_df


def create_attrs_files():
    finetuned = [False,True]
    output_folder = os.getenv("ATTRS_RESULTS")
    os.makedirs(output_folder, exist_ok=True)
    
    for model in models:
        for dataset in datasets:
            os.makedirs(os.path.join(output_folder, dataset), exist_ok=True)
            for finetuned_set in finetuned:
                finetuned_postfix = "_finetuned" if finetuned_set else ""
                print(f"\n{b_}Working with dataset {dataset} and model {model}{finetuned_postfix}{sr_}")
                result_prefix = os.path.join(CAPTIONS_DIR, dataset, f"{dataset}_{model}{finetuned_postfix}")
                res_file_path = f"{result_prefix}.csv"
                df = pd.read_csv(res_file_path)
                df = df.rename(columns={"res": "text"})
                df = add_attrs_column_batch(df) if torch.cuda.is_available() else add_attrs_column(df)
                attrs_dict = {}
                for index, row in df.iterrows():
                    file_name = row["id"]
                    # concatinate attributes with space
                    attributes = " ".join(set(row["attributes"]))
                    
                    if attributes == "":
                        print(f"Empty attributes for {file_name}")

                    attrs_dict[file_name] = {"caption": attributes}
                attrs_file = os.path.join(output_folder, dataset, f"{dataset}_{model}{finetuned_postfix}_attrs.json")
                with open(attrs_file, "w", encoding="utf-8") as f:
                    json.dump(attrs_dict, f, ensure_ascii=False)  # to include all character
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for scoring, accuracy, and MAP calculation.")
    parser.add_argument("mode", choices=["score", "accuracy", "map", "files"], help="Mode to run the tool")
    parser.add_argument("--model", help="Model name (required for single mode).")
    parser.add_argument("--dataset", help="Dataset name (required for single mode).")
    parser.add_argument("--full-run", action="store_false", help="Run the full accuracy evaluation (including predicting of classes using BERT Model).")
    parser.add_argument("--finetuned", action="store_true", help="Use a finetuned model.")

    args = parser.parse_args()

    if args.mode == "files":
        create_attrs_files()

    # Validation: if model is set, dataset must also be set (and vice versa)
    if (args.model and not args.dataset) or (args.dataset and not args.model):
        parser.error("Both --model and --dataset must be specified together.")

    # Determine whether to run default or single mode
    if args.model and args.dataset:
        if args.mode == "score":
            compute_score(args.model, args.dataset, finetuned=args.finetuned)
        elif args.mode == "accuracy":
            compute_category_accuracy(
                args.model,
                args.dataset,
                finetuned=args.finetuned,
                full_run=args.full_run,
            )
        elif args.mode == "map":
            compute_map(args.model, args.dataset, finetuned=args.finetuned)
    else:
        # Default behavior run all models and datasets
        if args.mode == "score":
            score_mode(finetuned=args.finetuned)
        elif args.mode == "accuracy":
            accuracy_mode(finetuned=args.finetuned, full_run=args.full_run)
        elif args.mode == "map":
            map_mode(finetuned=args.finetuned)