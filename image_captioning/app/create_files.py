import argparse
import os
import json
import pandas as pd
import sys

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.pyevalcap.scorer import Scorer
from src.pyevalcap.map.map import Map
from src.pyevalcap.accuracy.accuracy import Accuracy, compute_accuracy

from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")

models = [
    "blip2-opt-2.7b",
    "blip2-opt-6.7b", 
    "blip2-flan-t5-xl", 
    "blip2-flan-t5-xxl",
    "llava-1.5-7b-hf",
    "llava-1.5-13b-hf"]

datasets = ["hm", "fic"]

finetuned_postfixes = ["", "_finetuned"]

def round_value(value):
    return round(round(value, 3) * 100, 1)

def load_json(file_path):
    with open(file_path, "r") as j:
        file = json.load(j)
    return file

hm_cat_path = os.getenv("HM_CATES_MAP")
hm_cat_map = load_json(hm_cat_path)
fic_cat_path = os.getenv("FIC_CATES_MAP")
fic_cat_map = load_json(fic_cat_path)

def get_category(dataset:str, number:int):
    """
    Converts a number to its corresponding category.

    :param dataset: str - The dataset mapping to use.
    :param number: int - The number to convert.
    :return: str - The category name, or a message if the number is not valid.
    """
    map = hm_cat_map if dataset == "hm" else fic_cat_map
    number_to_category = {str(v): k for k, v in map.items()} if dataset == "hm" else map
    return number_to_category.get(str(number), "Invalid number")

def create_files(num_samples=50, device="cuda"):
    print(f"Creating {num_samples} sample metrics results...")
    
    captions_path = os.getenv("CAPTION_RESULTS")

    results = pd.DataFrame()
    map = Map()
    scorer = Scorer()

    for dataset in datasets:
        gts_attrs_path = os.getenv("HM_ATTRS") if dataset == "hm" else os.getenv("FIC_ATTRS")
        attrs_path = os.getenv("HM_ATTRS_TRAIN") if dataset == "hm" else os.getenv("FIC_ATTRS_ALL") 
        gts_attrs = load_json(gts_attrs_path)
        
        cates_file = os.getenv("HM_CATES") if dataset == "hm" else os.getenv("FIC_CATES")
        gts_cates = load_json(cates_file)
        attrs = load_json(attrs_path)

        acc = Accuracy(dataset, device)
        
        for model_name in models:
            for finetuned_postfix in finetuned_postfixes:
                result_prefix = os.path.join(captions_path, f"{dataset}_results", f"{dataset}_{model_name}{finetuned_postfix}")
                gts_file_path = f"{result_prefix}_gts.json"
                res_file_path = f"{result_prefix}_res.json"

                with open(gts_file_path, "r") as gts_file:
                    GT = json.load(gts_file)

                with open(res_file_path, "r") as res_file:
                    RES = json.load(res_file)

                IDs = list(GT.keys())[:num_samples]

                for id, idx in zip(IDs, range(num_samples)):
                    print(f"Processing image id {id}...")
                    gts = {id: GT[id]}
                    pred = {id: RES[id]}

                    metrics = scorer.evaluate(gts, pred, [id])

                    prec, reca, true_p, selected = map.calculate_pr(RES[id], gts_attrs[id], attrs)
                    print(f"Precision: {prec}, Recall: {reca}")

                    metrics["model"] = f"{model_name}{finetuned_postfix}"
                    metrics["dataset"] = dataset

                    metrics["image_id"] = id
                    metrics["gt_caption"] = GT[id][0]["caption"]

                    
                    metrics["gt_attrs"] = gts_attrs[id]["caption"]
                    metrics["attrs_selected"] = selected
                    metrics["true_positives"] = true_p
                    metrics["pred_caption"] = RES[id][0]["caption"]

                    metrics["precision"] = prec
                    metrics["recall"] = reca

                    pred_cat = acc.compute_class(pred)
                    gts_cat  = gts_cates[idx]

                    metrics["pred_cat"] = get_category(dataset, pred_cat[0])
                    metrics["gt_cat"] = get_category(dataset, gts_cat)

                    metrics_df = pd.DataFrame([metrics])
                    results = pd.concat([results, metrics_df], ignore_index=True)

        del acc

    # List of columns to apply the function to
    metrics_cols = [
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
        "CIDEr",
        "SPICE",
    ]

    # Apply the round function to the subset of columns
    results[metrics_cols] = results[metrics_cols].apply(round_value)

    results.to_csv(f"metrics_results_{num_samples}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models and save sample metrics results to a CSV file.")
    parser.add_argument("--device",default="cuda", help="Device to use for evaluation.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to create metrics results for.")
    args = parser.parse_args()

    create_files(num_samples=args.num_samples,device=args.device)
