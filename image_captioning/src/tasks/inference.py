import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import os
import json

# Own modules
import sys

sys.path.append("..")
sys.path.append("../dataset")
from dataset.dataset import get_datasets
from models.models import BLIP2Model, LlavaModel

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
CAPTIONS_DIR = os.path.join(RESULTS_DIR, "caption_results")


def extract_caption_after_prompt(caption):
    index = caption.find("ASSISTANT:")
    if index != -1:
        result = caption[index + len("ASSISTANT:") :]
        return result.strip()
    else:
        return "No caption found after 'ASSISTANT:'"


def save_to_csv(foldername, filename, res_list, gts_list, ids_list):
    data = {"res": res_list, "gts": gts_list, "id": ids_list}

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(foldername, f"{filename}.csv"), index=False)


def save_to_json(foldername, filename, res_list, gts_list, ids_list):
    gts_dict = {}
    res_dict = {}
    for img, gt, res in zip(ids_list, gts_list, res_list):
        img_key = img.filename if hasattr(img, "filename") else img
        if img_key not in gts_dict:
            gts_dict[img_key] = []
            res_dict[img_key] = []
        gts_dict[img_key].append({"caption": gt})
        res_dict[img_key].append({"caption": res})

    with open(os.path.join(foldername, f"{filename}_gts.json"), "w", encoding="utf-8") as gts_json_file:
        json.dump(gts_dict, gts_json_file, indent=2, ensure_ascii=False)

    with open(os.path.join(foldername, f"{filename}_res.json"), "w", encoding="utf-8") as res_json_file:
        json.dump(res_dict, res_json_file, indent=2, ensure_ascii=False)


def run_inference(model, dataloader, processor):
    res_list = []
    gts_list = []
    ids_list = []

    for batch in tqdm(dataloader, desc="Processing dataset", unit="batch", file=sys.stdout):
        inputs = batch.pop("inputs").to(model.device)
        gts = batch.pop("text")
        id = batch.pop("id")

        generated_ids = model.model.generate(**inputs, max_length=100, do_sample=False)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        if model.__class__.__name__ == "LlavaModel":
            res_list.extend([extract_caption_after_prompt(res.strip()) for res in generated_text])
        else:
            res_list.extend([res.strip() for res in generated_text])
        gts_list.extend(gts)
        ids_list.extend(id)

        sys.stdout.flush()

    return res_list, gts_list, ids_list


def create_inference(
    base_checkpoint: str,
    adapter_checkpoint: str = None,
    save_json: bool = True,
    save_csv: bool = False,
    batch_size: int = 32,
    dataset: str = "hm",
    device: str = "cuda",
    debug: bool = False,
):

    test_dataset = get_datasets(dataset, test=True, debug=debug)

    if "blip2" in base_checkpoint:
        model = BLIP2Model(
            base_checkpoint=base_checkpoint,
            adapter_checkpoint=adapter_checkpoint,
            device=device,
        )
    elif "llava" in base_checkpoint:
        model = LlavaModel(
            base_checkpoint=base_checkpoint,
            adapter_checkpoint=adapter_checkpoint,
            device=device,
        )

    model_class = model.__class__.__name__
    model_name = model.model.config.name_or_path.split("/")[-1]
    tokenizer = model.tokenizer
    processor = model.processor

    def collator(batch):
        processed_batch = {}
        imgs = [item["image"] for item in batch]
        processed_batch["text"] = [item["text"] for item in batch]
        processed_batch["id"] = [item["id"] for item in batch]
        if model_class == "BLIP2Model":
            processed_batch["inputs"] = processor(imgs, return_tensors="pt")
        elif model_class == "LlavaModel":
            tokenizer.padding_side = "left"  # recommended by huggingface
            processed_batch["inputs"] = processor([model.prompt] * len(imgs), imgs, return_tensors="pt")
        return processed_batch

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    print(f"Using model from {model_name} and {processor.__class__.__name__} processor for {dataset}")

    res_list, gts_list, ids_list = run_inference(model, dataloader, processor)

    finetuned = "_finetuned" if adapter_checkpoint else ""
    filename = f"{dataset}_{model_name}{finetuned}"
    foldername = os.path.join(CAPTIONS_DIR, f"{dataset}")

    os.makedirs(foldername, exist_ok=True)
    if save_csv:
        save_to_csv(foldername, filename, res_list, gts_list, ids_list)
    if save_json:
        save_to_json(foldername, filename, res_list, gts_list, ids_list)

    if save_csv or save_json:
        print(f"Results saved in {foldername}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified model and dataset")
    parser.add_argument("--base_checkpoint",type=str,required=True,help="Path to the base model checkpoint",)
    parser.add_argument("--adapter_checkpoint",type=str,default=None,help="Path to the adapter checkpoint (optional)",)
    parser.add_argument("--save_json",type=str,default=None,help="Flag to save the generated text as json",)
    parser.add_argument("--save_csv",type=str,default=None,help="Flag to save the generated text as json",)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--dataset",type=str,required=True,choices=["fic", "hm"],help="Dataset to use for inference (fic or hm)",)
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu",help="Device to run inference on (cuda/cpu)",)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        CAPTIONS_DIR = os.path.join(CAPTIONS_DIR, "debug")

    base_checkpoint = os.path.join(ROOT_DIR, "models", args.base_checkpoint)
    if args.adapter_checkpoint:
        adapter_checkpoint = os.path.join(ROOT_DIR, "finetuned_models", args.adapter_checkpoint)

    print(f"Running inference with following parameters:")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Adapter checkpoint: {args.adapter_checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")

    create_inference(
        base_checkpoint,
        adapter_checkpoint,
        args.save_json,
        args.save_csv,
        args.batch_size,
        args.dataset,
        args.device,
        args.debug,
    )
