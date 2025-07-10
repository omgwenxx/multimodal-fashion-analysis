from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
import torch
import json
import os
import pandas as pd
import argparse
from huggingface_hub import PyTorchModelHubMixin

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

from dotenv import load_dotenv
load_dotenv()
HM_PATH = os.getenv("HM_PROCESSED")
FIC_PATH = os.getenv("FACAD")
MODEL_DIR = "./models"

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


set_seed(42)


def get_hm_labels():
    categories = [
        "Bra",
        "Swimwear bottom",
        "Pyjama set",
        "Underwear bottom",
        "T-shirt",
        "Bracelet",
        "Sweater",
        "Shorts",
        "Blouse",
        "Vest top",
        "Trousers",
        "Top",
        "Belt",
        "Dress",
        "Pyjama jumpsuit/playsuit",
        "Jacket",
        "Hoodie",
        "Skirt",
        "Other shoe",
        "Scarf",
        "Swimwear top",
        "Socks",
        "Jumpsuit/Playsuit",
        "Garment Set",
        "Boots",
        "Pyjama bottom",
        "Hair/alice band",
        "Shirt",
        "Hair clip",
        "Other accessories",
        "Polo shirt",
        "Earring",
        "Underwear Tights",
        "Leggings/Tights",
        "Blazer",
        "Sandals",
        "Coat",
        "Dungarees",
        "Hat/beanie",
        "Sneakers",
        "Swimsuit",
        "Cardigan",
        "Night gown",
        "Bikini top",
        "Cap/peaked",
        "Sarong",
        "Bodysuit",
        "Ballerinas",
        "Bag",
        "Gloves",
        "Flip flop",
        "Swimwear set",
        "Hair ties",
        "Outdoor Waistcoat",
        "Sunglasses",
        "Wallet",
        "Necklace",
        "Hat/brim",
        "Ring",
        "Watch",
        "Wedge",
        "Costumes",
        "Heeled sandals",
        "Hair string",
        "Underwear body",
        "Pumps",
        "Outdoor trousers",
        "Slippers",
        "Tailored Waistcoat",
        "Felt hat",
        "Robe",
        "Tie",
        "Flat shoe",
        "Beanie",
        "Heels",
        "Outdoor overall",
        "Weekend/Gym bag",
        "Kids Underwear top",
        "Earrings",
        "Long John",
        "Bootie",
        "Flat shoes",
        "Underdress",
        "Underwear set",
        "Cap",
        "Leg warmers",
        "Underwear corset",
        "Bucket hat",
        "Accessories set",
    ]
    labels = {category: index for index, category in enumerate(categories)}
    return labels


class EvalDataset(Dataset):
    def __init__(self, res):
        self.captions = [res[id][0]["caption"] for id in res]

    def __getitem__(self, idx):
        return self.captions[idx]

    def __len__(self):
        return len(self.captions)


class Accuracy:
    def __init__(self, dataset, device="cpu"):
        assert dataset.lower() in {"fic", "hm"}

        cwd = os.path.dirname(os.path.abspath(__file__))
        model_temp_dir = os.path.join(cwd, MODEL_DIR)
        self.device = device
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.num_classes = 89 if dataset == "hm" else 78
        
        print(f"Initializing bert model for {dataset} dataset with {self.num_classes} classes on {'GPU' if 'cuda' in device else 'CPU' }...")
        
        self.model = BertClassifier(dataset, self.num_classes)
        model_path = "hm-bestacc-ep15-bs128.pth" if dataset == "hm" else "fic-bestacc-ep5-bs128.pth"
        self.model.load_state_dict(torch.load(os.path.join(model_temp_dir,model_path)))

    def compute_class(self, ref_data, batch_size=64):

        def collate_fn(batch):
            texts = self.tokenizer(batch, padding='max_length', max_length = 512, truncation=True,return_tensors="pt")

            return {
                "input_ids": texts["input_ids"],
                "attention_mask": texts["attention_mask"],
            }

        eval_ds = EvalDataset(ref_data)
        dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        model = self.model.to(self.device)
        predictions = []

        with torch.no_grad():
            for test_input in tqdm(dataloader, desc="Processing ref data", mininterval=40):
                attention_mask = test_input["attention_mask"].to(self.device)
                input_ids = test_input["input_ids"].to(self.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.extend(output.argmax(dim=1).tolist())

        return predictions

    def method(self):
        return "Accuracy"


class FICCategoryDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train', 'val', or 'test'
        """
        self.split = split.upper()
        assert self.split in {"TRAIN", "VAL", "TEST"}
        print(f"Loading {split.lower()} split for FACAD")

        with open(os.path.join(FIC_PATH, self.split + "_CAPTIONS_RAW" + ".json"), "r") as j:
            self.captions = json.load(j)

        with open(os.path.join(FIC_PATH, self.split + "_CATES_FLAT.json"), "r") as j:
            self.categories = json.load(j)
            self.num_classes = len(set(self.categories))

    def __getitem__(self, i):
        """
        :returns: Properties of ith item as Tuple
        :rtype: Tuple (Tensor, Tensor)
        """
        text = self.captions[i]
        category = self.categories[i]
        return {"text": text, "category": category}

    def __len__(self):
        return len(self.captions)


class HMCategoryDataset(Dataset):
    def __init__(self, df):
        labels = get_hm_labels()
        df = df.rename(columns={"product_type_name": "category"})
        self.labels = [labels[label] for label in df["category"]]
        self.texts = df["text"]
        self.num_classes = len(labels.keys())

    def classes(self):
        return self.labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        category = self.labels[idx]
        return {"text": text, "category": category}

    def __len__(self):
        return len(self.labels)


class BertClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, dataset: str, num_classes, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.model_name = "bert-base-uncased"
        print(f"Loading BERT model {self.model_name} for {dataset} dataset...")

        self.bert = BertModel.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)  # in features, out features = number of classes
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(
    model,
    train_dataloader,
    val_dataloader,
    learning_rate,
    epochs,
    dataset,
    device="cuda",
):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Using device:", device)
    model = model.to(device)
    criterion = criterion.to(device)

    print(f"Starting training with {len(train_dataloader.dataset)} train samples and {len(val_dataloader.dataset)} val samples...")

    best_acc = 0
    train_batch_size = train_dataloader.batch_size

    for epoch_num in range(1, epochs + 1):
        total_acc_train = 0
        total_loss_train = 0

        for batch in tqdm(train_dataloader):
            train_label = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)

            output = model(input_ids, attention_mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for batch in val_dataloader:

                val_label = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)
                input_id = batch["input_ids"].to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()

                total_acc_val += acc

            val_epoch_acc = total_acc_val / len(val_dataloader.dataset)
            print
            if val_epoch_acc > best_acc:
                print(
                    f"{b_}New best accuracy found:{sr_}",
                    val_epoch_acc,
                    "old best:",
                    best_acc,
                )
                best_acc = total_acc_val / len(val_dataloader.dataset)
                torch.save(
                    model.state_dict(),
                    f"{MODEL_DIR}/{dataset}-bestacc-ep{epoch_num}-bs{train_batch_size}.pth",
                )
                print(f"{b_}Save model with best accuracy so far:{sr_}", best_acc)

        print(
            f"Epochs: {epoch_num} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}"
        )

    return best_acc


def evaluate(model, test_dataloader, device="cuda"):

    print("Using device:", device)
    model = model.to(device)

    total_acc_test = 0
    with torch.no_grad():

        for batch in tqdm(test_dataloader, desc="Processing test data"):

            test_label = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)

            output = model(input_ids, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_dataloader.dataset): .3f}")


def get_data(dataset: str, debug: bool = False):
    if dataset == "fic":
        print("Loading FIC data...")
        train_data = FICCategoryDataset("train")
        val_data = FICCategoryDataset("val")
        test_data = FICCategoryDataset("test")
    elif dataset == "hm":
        print("Loading HM data...")
        train_df = pd.read_csv(HM_PATH + "/hm_train.csv")
        val_df = pd.read_csv(HM_PATH + "/hm_val.csv")
        test_df = pd.read_csv(HM_PATH + "/hm_test.csv")

        train_data = HMCategoryDataset(train_df)
        val_data = HMCategoryDataset(val_df)
        test_data = HMCategoryDataset(test_df)

    if debug:
        num = 100
        train_data = Subset(train_data, range(num))
        val_data = Subset(val_data, range(num))
        test_data = Subset(test_data, range(num))

    return train_data, val_data, test_data


def create_setup(
    dataset,
    batch_size=64,
    epochs=15,
    learning_rate=1e-5,
    device="cuda",
    debug: bool = False,
):
    if debug:
        print(f"{y_}Running in Debug Mode{sr_}")

    train_data, val_data, test_data = get_data(dataset, debug)

    num_classes = train_data.dataset.num_classes if debug else train_data.num_classes  # should be 78 for FIC and 89 for HM
    print("\nInitializing model with", num_classes, "classes...")
    model = BertClassifier(dataset=dataset, num_classes=num_classes)
    tokenizer = BertTokenizer.from_pretrained(model.model_name)

    def collate_fn(batch):
        # Extract text and category from the batch
        texts = [item["text"] for item in batch]
        categories = [item["category"] for item in batch]

        # Tokenize the texts
        encoded_texts = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        labels = torch.tensor(categories, dtype=torch.long)

        # Return processed data
        return {
            "input_ids": encoded_texts["input_ids"],
            "attention_mask": encoded_texts["attention_mask"],
            "labels": labels,
        }

    print("Creating dataloaders with batch size", batch_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"\nStarting training with LR={learning_rate}, EPOCHS={epochs}, BATCH_SIZE={batch_size}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = train(model, train_dataloader, val_dataloader, learning_rate, epochs, dataset, device)
    print(f"\n{b_}Best Accuracy: {best_acc:.3f}{sr_}")

    print("\nLoading test data and starting evaluation...")
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    evaluate(model, test_dataloader, device)


def compute_accuracy(true_classes, predicted_classes):
    """
    Compute the accuracy between two lists of classes.

    Parameters:
    true_classes (list): List of true class labels.
    predicted_classes (list): List of predicted class labels.

    Returns:
    float: Accuracy score, which is the proportion of correct predictions.
    """
    # Ensure that the lists have the same length
    if len(true_classes) != len(predicted_classes):
        raise ValueError("Lists must have the same length")

    # Count the number of matching elements
    num_correct = sum(1 for true, pred in zip(true_classes, predicted_classes) if true == pred)

    # Compute accuracy
    accuracy = num_correct / len(true_classes)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning Accuracy")
    parser.add_argument("dataset", choices=["fic", "hm"], help="Dataset to run (fic or hm)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="GPU device")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    create_setup(args.dataset, args.batch_size, args.epochs, args.lr, args.device, args.debug)
