import os
import torch
import numpy as np
import argparse
import pandas as pd
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

# Load environment variables
load_dotenv()

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Extract BLIP2 features from images.")
parser.add_argument("--reduce_dim", "-rd", action="store_true", help="Reduce dimension by averaging vision features.")
parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing even if output folders exist.")
parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batchsize for dataloader")
args = parser.parse_args()

# Define paths with dynamic folder naming
CSV_FILE = os.getenv("HM_FILTERED_ARTICLES")
BLIP_FOLDER = "blip2_features"
DIM_POSTFIX = "reduce_dim" if args.reduce_dim else "full_dim"

VISION_OUTPUT_FOLDER = os.path.join(os.getenv("HM_RAW"), BLIP_FOLDER, f"visual_features_{DIM_POSTFIX}")
QFORMER_OUTPUT_FOLDER = os.path.join(os.getenv("HM_RAW"), BLIP_FOLDER, f"qformer_features_{DIM_POSTFIX}")

# Check if features already exist
if not args.force and os.path.exists(VISION_OUTPUT_FOLDER) and os.path.exists(QFORMER_OUTPUT_FOLDER):
    print(f"📢 Feature extraction skipped: Output folders already exist.")
    print(f"   If you want to re-run, use the '--force' or '-f' flag.")
    exit(0)  # Exit script if folders exist and --force is not provided

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load base model and processor
def load_models():
    base_model_path = f"{os.getenv('ROOT_DIR')}/models/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(base_model_path)
    basic_model = Blip2ForConditionalGeneration.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    adapter_model_path = f"{os.getenv('ROOT_DIR')}/blip2-opt-2.7b-hm/bestloss-ep8-blip2-opt-2.7b-hm-all_LoRA-lr5e-5-bs16"
    peft_model = PeftModel.from_pretrained(basic_model, model_id=adapter_model_path).to(device)
    peft_model.eval()
    
    return processor, peft_model

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["img_path"]
        detail_desc = row["detail_desc"]
        image_name = os.path.basename(image_path)
        image = Image.open(f"{os.getenv("HM_RAW")}/images_flat/{image_name}").convert("RGB")
        
        return image, detail_desc, image_name

# Main function
def main():
    df = pd.read_csv(CSV_FILE)

    os.makedirs(VISION_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(QFORMER_OUTPUT_FOLDER, exist_ok=True)

    dataset = ImageDataset(df)
    processor, peft_model = load_models()

    def collate_fn(batch):
        images, descriptions, image_names = zip(*batch)
        inputs = processor(images=images, text=descriptions, return_tensors="pt", padding=True)

        return inputs, list(image_names)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for batch in tqdm(dataloader):        
        inputs, image_names = batch
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = peft_model(**inputs, output_hidden_states=True)
            vision_features = outputs["vision_outputs"]["last_hidden_state"]  # Shape (batch, 257, 1408)
            qformer_features = outputs["qformer_outputs"]["last_hidden_state"]  # Shape (batch, 32, 768)

            if args.reduce_dim:
                vision_features = torch.mean(vision_features, dim=1)  # Shape (batch, 1408)
            else:
                vision_features = vision_features.view(vision_features.shape[0], -1)  # Flatten to (batch, 257 * 1408)

            vision_features = vision_features.float().cpu().numpy()
            qformer_features = qformer_features.view(qformer_features.shape[0], -1).float().cpu().numpy()

        # Save each image's feature vector as a separate `.npy` file
        for i in range(len(image_names)):  
            img_name = image_names[i]
            v_feat = vision_features[i]
            q_feat = qformer_features[i]

            vision_output_path = os.path.join(VISION_OUTPUT_FOLDER, f"{os.path.splitext(img_name)[0]}.npy")
            qformer_output_path = os.path.join(QFORMER_OUTPUT_FOLDER, f"{os.path.splitext(img_name)[0]}.npy")

            np.save(vision_output_path, v_feat)
            np.save(qformer_output_path, q_feat)

            print(f"✅ Saved features for {img_name}")

    print(f"🎉 Feature extraction completed! Results saved in:")
    print(f"   - {VISION_OUTPUT_FOLDER}")
    print(f"   - {QFORMER_OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
