import os
import gc
import copy
import time
from typing import List

# For data manipulation
import numpy as np

# Pytorch Imports
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# Utils
from tqdm import tqdm
from collections import defaultdict

# For Transformer Models
from transformers import AdamW, set_seed

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")
import wandb

import argparse
import subprocess
from numba import cuda

import sys

sys.path.append("..")
sys.path.append("../dataset")
from dataset.dataset import get_datasets
from models.models import get_lora_model, get_model
from torch.utils.data import Dataset, DataLoader
import torch

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")


def set_seed(seed: int = 42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    print(f"Setting seed for reproducibility to {seed}...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set a fixed value for the hash seed
    


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"],
            text=item["text"],
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


def check_batch_size(model, dataset, device: str = "cpu", debug: bool = False, train_batch_size: int = 2):
    def bytes_to_gb(bytes):
        gb = bytes / 1024**3
        return gb

    still_try = True
    while still_try:
        try:
            script_name = "python -m train.train"
            model_arg = f"--checkpoint {model}"
            dataset_arg = f"--dataset {dataset}"
            device_arg = f"--device {device}"
            debug_args = f"--debug" if debug else ""
            epochs_arg = f"--epochs 15" if debug else "--epochs 5"
            train_batch_arg = f"--train_batch_size {train_batch_size}"
            command = " ".join(
                [
                    script_name,
                    model_arg,
                    dataset_arg,
                    device_arg,
                    train_batch_arg,
                    epochs_arg,
                    debug_args,
                ]
            )
            print(f"Running command: {command}")
            # Print memory usage
            mem_info = torch.cuda.mem_get_info(device)

            total_memory = bytes_to_gb(mem_info[1])
            free_memory = bytes_to_gb(mem_info[0])

            print(f"Total GPU memory: {total_memory:.2f} GB")
            print(f"Available GPU memory: {free_memory:.2f} GB")

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, text=True)
            text, err = process.communicate()

            # if result.stderr has cuda out of memory error, reduce batch size by 2 and retry
            if "CUDA out of memory" in err:
                print(f"{b_}CUDA out of memory error encountered.{sr_}")
                try:
                    print("Killing process to free up resources...")
                    process.kill()
                    return_code = process.wait()
                    del process
                    gc.collect()
                    print(f"Process terminated with return code: {return_code}")
                except OSError:
                    # can't kill a dead proc
                    pass
                delay = 1
                if train_batch_size <= 2:
                    print("Last batch size then aborting.")
                    train_batch_size = 1
                    still_try = False
                train_batch_size = train_batch_size - 2
                print(f"Retrying in {delay} seconds...\n")
                time.sleep(delay)
            else:
                print(
                    f"\n{y_}Final batch size:",
                    train_batch_size,
                    "with model:",
                    f"{model.split('/')[-1]}",
                    f"{sr_}\n",
                )
                still_try = False
        except subprocess.CalledProcessError as e:
            if "CUDA out of memory" in e.stderr:
                print("CUDA out of memory error encountered.")
                delay = 1
                train_batch_size = train_batch_size - 2
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("An error occurred that is not related to CUDA out of memory.")
                print(f"Error: {e}")
                raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


def tensor_to_img(tensor):
    transform = T.ToPILImage()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = transform(tensor)
    return img


def log_image_table(images, predicted, labels, epoch):
    "Log a wandb.Table with (img, pred, target)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target", "epoch"])
    for img, logits, targ in zip(images.cpu(), predicted.cpu(), labels.cpu()):
        img = wandb.Image(tensor_to_img(img))
        pred_ids = F.softmax(logits, dim=-1).argmax(dim=-1)
        pred = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
        targ = model.tokenizer.decode(targ, skip_special_tokens=True)
        table.add_data(img, pred, targ, epoch)
    wandb.log({f"predictions_table_ep{epoch}": table})


def fetch_scheduler(optimizer):
    if CONFIG["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["T_max"], eta_min=CONFIG["min_lr"])
    elif CONFIG["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG["T_0"], eta_min=CONFIG["min_lr"])
    elif CONFIG["scheduler"] == None:
        return None

    return scheduler


def format_lr(lr: float):
    return format(lr, ".0e").replace("e-0", "e-").replace("e+0", "e+")


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device)
        pixel_values = data["pixel_values"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

        loss = outputs.loss
        loss = loss / CONFIG["n_accumulate"]
        loss.backward()

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device)
        pixel_values = data["pixel_values"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

        if step == 0:
            log_image_table(pixel_values, outputs.logits, input_ids, epoch)

        loss = outputs.loss

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"])

    gc.collect()

    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs, patience):
    wandb.watch(model, log_freq=100)  # To automatically log gradients

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print("[INFO] Using GPU: {} | {}\n".format(gpu_name, device))

    start = time.time()
    best_epoch_loss = np.inf
    history = defaultdict(list)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
        )

        val_epoch_loss = valid_one_epoch(model, valid_loader, device=device, epoch=epoch)

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)

        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            run.summary["Best Loss"] = best_epoch_loss

            model_dir = f"{CONFIG['model_name']}-{CONFIG['dataset']}"
            os.makedirs(os.path.join(OUTPUT_DIR, model_dir), exist_ok=True)

            best_model_path = os.path.join(
                OUTPUT_DIR,
                model_dir,
                f"bestloss-ep{epoch}-{CONFIG['model_name']}-{CONFIG['dataset']}-{'all_LoRA' if lora_layers == 'all-linear' else 'QV_LoRA'}-lr{format_lr(CONFIG['learning_rate'])}-bs{CONFIG['train_batch_size']*CONFIG['n_accumulate']}",
            )
            model.save_pretrained(best_model_path)
            print(f"Model Saved{sr_}")
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
            if patience_counter > patience - 1:
                print(f"Early stopping, no improvement after {patience} epochs")
                break  # early stopping, patience of 3

        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    # model.load_state_dict(torch.load(best_model_path)) # torch interface
    model.from_pretrained(model.get_base_model(), best_model_path)  # hugggingface interface

    return model, history


def create_training(
    checkpoint: str,
    dataset: str = "hm",
    device: str = "cpu",
    debug: bool = False,
    train_batch_size: int = 4,
    valid_batch_size: int = 8,
    learning_rate: float = 5e-4,
    n_accumulate: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_layers: str | List[str] = "all-linear",  # Accepts either a string or a list of strings
    epochs: int = 10,
    patience: int = 3,
):
    global CONFIG, model, train_loader, valid_loader, optimizer, scheduler, run, OUTPUT_DIR
    models_output_dir = os.getenv("MODELS_OUTPUT_DIR")
    OUTPUT_DIR = f"{ROOT_DIR}/{models_output_dir}_debug" if debug else f"{ROOT_DIR}/{models_output_dir}"
    CONFIG = {
        "epochs": epochs,
        "model_name": f"{checkpoint.split('/')[-1]}",
        "dataset": dataset,
        "train_batch_size": train_batch_size,
        "valid_batch_size": valid_batch_size,
        "learning_rate": learning_rate,
        "scheduler": "CosineAnnealingLR",
        "min_lr": 1e-6,
        "T_max": 500,
        "weight_decay": 1e-6,
        "n_accumulate": n_accumulate,
        "device": device,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_layers": lora_layers,
        "debug": debug,
    }

    # Initialize W&B run
    run = wandb.init(
        project="master-thesis-debug" if debug else "master-thesis",  # name of project
        config=CONFIG,  # metadata
        job_type="Train",
        tags=[
            CONFIG["model_name"],
            CONFIG["dataset"],
            f"lr{format_lr(CONFIG['learning_rate'])}",
            f"bs{CONFIG['train_batch_size']*CONFIG['n_accumulate']}",
            f"{'all_LoRA' if lora_layers == 'all-linear' else 'QV_LoRA'}",
        ],
        name=f"{CONFIG['model_name']}-{CONFIG['dataset']}-{'all_LoRA' if lora_layers == 'all-linear' else 'QV_LoRA'}-lr{format_lr(CONFIG['learning_rate'])}-bs{CONFIG['train_batch_size']*CONFIG['n_accumulate']}",
    )  # name of run

    if debug:
        print(f"{y_}Running in Debug Mode{sr_}")

    print(
        f"{b_}Model: {CONFIG['model_name']} | Dataset: {dataset} | Device: {device} | Debug: {debug} | Epochs: {epochs} | Train Batch Size: {train_batch_size} | Learning Rate: {format_lr(CONFIG['learning_rate'])} | N Accumulate: {n_accumulate}{sr_}"
    )
    print(f"{b_}Lora Alpha: {lora_alpha} | Lora Rank: {lora_rank} | Lora Layers: {'all_LoRA' if lora_layers == 'all-linear' else 'QV_LoRA'}{sr_}")
    model = get_model(checkpoint, device=CONFIG["device"])
    peft_model = get_lora_model(
        model.model,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        lora_layers=CONFIG["lora_layers"],
    )

    print()
    train_ds, val_ds = get_datasets(dataset, debug=debug)
    train_dataset = ImageCaptioningDataset(train_ds, model.processor)
    valid_dataset = ImageCaptioningDataset(val_ds, model.processor)

    print(f"Train Dataset Size: {len(train_dataset)} | Validation Dataset Size: {len(valid_dataset)}")

    peft_model.to(CONFIG["device"])

    optimizer = AdamW(
        peft_model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = fetch_scheduler(optimizer)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG["train_batch_size"])
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=CONFIG["valid_batch_size"])

    print()
    peft_model, history = run_training(
        peft_model,
        optimizer,
        scheduler,
        patience=patience,
        device=CONFIG["device"],
        num_epochs=CONFIG["epochs"],
    )

    run.finish()
    print()
    wandb.finish()

    # Clean up
    del model, history, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=device)
    device = cuda.get_current_device()
    device.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for running training.")
    parser.add_argument("--run", type=str, default="train", help="Run type, e.g., train or test")
    parser.add_argument("--model", type=str, required=True, help="Model variant")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--checkpoint_model_id", type=str, default="-1", help="Checkpoint model ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--device", type=str, default="cuda", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Effective batch size, used to calculate gradient accumulation steps",
    )
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Scheduler type")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument(
        "--T_max",
        type=int,
        default=500,
        help="Maximum number of iterations for the scheduler",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimization")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="LoRA layers to train [all-linear, None]",
    )

    args = parser.parse_args()

    checkpoint = f"{ROOT_DIR}/models/{args.model}"
    dataset = args.dataset
    learning_rate = args.learning_rate
    n_accumulate = args.batch_size // args.train_batch_size or 1  # effective batch size divided by train batch size to set n_accumulate
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    epochs = args.epochs
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_layers = args.lora_layers
    device = args.device
    debug = args.debug  # the value of the argument to True if the flag is provided, and False if it is not

    create_training(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        n_accumulate=n_accumulate,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        lora_alpha=lora_alpha,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        lora_layers=lora_layers if lora_layers == "all-linear" else None,
        debug=debug,
    )
