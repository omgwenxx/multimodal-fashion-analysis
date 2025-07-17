import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import stanza  # downloads english package per default
import re
import time
import ast
import json
import torch
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import DatasetDict, Dataset, Image
import shutil
import argparse
from dotenv import load_dotenv

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger("preprocessing.preprocess_hm")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
HM_PATH = os.getenv("HM_RAW")
HM_PATH_PROCESSED = os.getenv("HM_PROCESSED")
IMAGES_PATH = os.getenv("HM_IMAGES")
HM_SPLIT_FOLDER = os.getenv("HM_SPLIT_FOLDER")


def simplify_color(color: str) -> str:
    """
    Simplifies a color string to one of the predefined color categories.

    Args:
        color (str): The color string to be simplified.

    Returns:
        str: The simplified color category.
    """
    color = color.lower()
    if color in ["black", "white", "grey", "silver", "gold", "transparent"]:
        return color
    elif "blue" in color:
        return "blue"
    elif "bronze" in color:
        return "bronze"
    elif "green" in color:
        return "green"
    elif "red" in color:
        return "red"
    elif "yellow" in color:
        return "yellow"
    elif "orange" in color:
        return "orange"
    elif "pink" in color:
        return "pink"
    elif "purple" in color:
        return "purple"
    elif "beige" in color:
        return "beige"
    elif "white" in color:
        return "white"
    elif "grey" in color:
        return "grey"
    elif "turquoise" in color:
        return "turquoise"
    else:
        return "other"


def quartiles(data: pd.Series) -> tuple[float, float, float]:
    """
    Calculate the quartiles of a given data series, returns 25%,50%,75% percentiles as tuple.
    """
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    return q1, q2, q3


def plot_categories(articles: pd.DataFrame) -> None:
    """
    Plot a histogram of the category counts
    """
    category_counts = articles["product_type_name"].value_counts()

    plt.figure(figsize=(10, 6))
    plt.hist(category_counts.values, color="skyblue", bins=50)  # Adjust the number of bins as needed
    plt.title("Histogram of Category Counts")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.show()

    q1, q2, q3 = quartiles(category_counts.values)
    print(f"Category Count Quartiles Q1: {q1}, Q2: {q2}, Q3: {q3}")


def filter_fashion_and_accessory_related(product_types: pd.Index) -> list[str]:
    """
    Gets a list of product types and filters out those that are not fashion or accessory related
    """
    fashion_keywords = {
        "Wedge",
        "Trousers",
        "Dress",
        "Sweater",
        "T-shirt",
        "Top",
        "Blouse",
        "Jacket",
        "Shorts",
        "Shirt",
        "Vest top",
        "Underwear bottom",
        "Skirt",
        "Hoodie",
        "Bra",
        "Socks",
        "Leggings/Tights",
        "Sneakers",
        "Cardigan",
        "Garment Set",
        "Swimwear bottom",
        "Bag",
        "Jumpsuit/Playsuit",
        "Pyjama set",
        "Blazer",
        "Bodysuit",
        "Bikini top",
        "Sandals",
        "Swimsuit",
        "Sunglasses",
        "Underwear Tights",
        "Coat",
        "Polo shirt",
        "Pyjama jumpsuit/playsuit",
        "Ballerinas",
        "Dungarees",
        "Slippers",
        "Pyjama bottom",
        "Heeled sandals",
        "Swimwear set",
        "Pumps",
        "Underwear body",
        "Night gown",
        "Flat shoe",
        "Outdoor Waistcoat",
        "Robe",
        "Outdoor trousers",
        "Flip flop",
        "Kids Underwear top",
        "Costumes",
        "Tailored Waistcoat",
        "Sarong",
        "Outdoor overall",
        "Swimwear top",
        "Bootie",
        "Long John",
        "Hair ties",
        "Heels",
        "Underdress",
        "Flat shoes",
        "Leg warmers",
        "Bucket hat",
        "Underwear corset",
        "Boots",
        "Tie",
        "Cap",
        "Other accessories",
        "Hat/brim",
        "Accessories set",
        "Other shoe",
        "Underwear set",
    }
    accessory_keywords = {
        "Earring",
        "Hat/beanie",
        "Necklace",
        "Hair/alice band",
        "Hair clip",
        "Ring",
        "Hair string",
        "Bracelet",
        "Scarf",
        "Cap/peaked",
        "Belt",
        "Gloves",
        "Wallet",
        "Watch",
        "Beanie",
        "Heels",
        "Earrings",
        "Felt hat",
        "Weekend/Gym bag",
    }

    filtered_product_types = set()
    for ptype in product_types:
        if any(keyword.lower() in ptype.lower() for keyword in fashion_keywords) or any(keyword.lower() in ptype.lower() for keyword in accessory_keywords):
            filtered_product_types.add(ptype)
    return list(filtered_product_types)


def filter_articles_by_categories(articles: pd.DataFrame) -> pd.DataFrame:
    """
    Filter articles by categories with at least 7 items
    """
    category_counts = articles["product_type_name"].value_counts()

    filtered_categories = category_counts[category_counts >= 7]

    logger.debug(f"{len(filtered_categories.index)} categories with at least 7 items: \n{filtered_categories.index}")

    return filtered_categories


def load_data() -> pd.DataFrame:
    """
    Load the raw data from file defined in .env file
    """
    articles = pd.read_csv(f"{HM_PATH}/articles.csv")
    logger.info(f"Loaded {len(articles)} articles with {len(articles.columns)} columns")
    return articles


def add_noun_adj_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column to the DataFrame containing the extracted lemmatized nouns and adjectives
    """
    nlp = stanza.Pipeline("en", verbose=False)

    def extract_noun_adj(sentence):
        try:
            sentence = re.sub(r"-", "", sentence)  # to merge words that are separated by hyphen

            doc = nlp(sentence)
            noun_adj_words = [
                word.lemma for sent in doc.sentences for word in sent.words if word.upos in ["NOUN", "ADJ", "PROPN"]
            ]  # Extract NOUN, ADJ, and PROPN words from the tokens and lemmatize them
            return noun_adj_words
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Original sentence: {sentence}")
            return []

    df["text"] = df["text"].astype(str)

    tqdm.pandas()
    df["attributes"] = df["text"].progress_apply(extract_noun_adj)
    return df


def preprocess_articles(save_file: bool = True, plot: bool = False) -> pd.DataFrame:
    """
    Runs entire preprocessing pipeline for filtering items by categories
    * keep categories that have at least 7 items
    * remove items that are not fashion or accessory related
    """
    articles = load_data()

    # remove missing description or missing image
    logger.info(f"Initial number of items {len(articles)}")
    logger.info("Removing items without description or image..")
    articles = articles.dropna(subset=["detail_desc"])  # remove rows with missing descriptions

    def create_img_folder(article_id):
        return f"0{str(article_id)[:2]}"

    def create_img_id(article_id):
        return f"0{str(article_id)}"

    articles["img_folder"] = articles["article_id"].apply(create_img_folder)
    articles["img_id"] = articles["article_id"].apply(create_img_id)
    articles["img_path"] = IMAGES_PATH + "/" + articles["img_folder"] + "/" + articles["img_id"] + ".jpg"

    # Remove items without image
    def image_exists(img_path):
        return os.path.exists(img_path)

    tqdm.pandas()
    articles = articles[articles["img_path"].progress_apply(image_exists)]
    logger.info(f"Removed items without description or image. New number of items {len(articles)}")

    if plot:
        plot_categories(articles)
    filtered_categories = filter_articles_by_categories(articles)
    filtered_product_types = filter_fashion_and_accessory_related(filtered_categories.index)
    logger.debug(f"Filtered categories: \n{filtered_product_types}")

    filtered_articles = articles[articles["product_type_name"].isin(filtered_product_types)]

    logger.info(f"Filtered {len(filtered_articles)} articles with {len(filtered_product_types)} categories")

    if save_file:
        if not os.path.exists(HM_PATH_PROCESSED):
            os.makedirs(HM_PATH_PROCESSED)

        filtered_articles.to_csv(os.path.join(HM_PATH_PROCESSED, "filtered_articles.csv"), index=False)
        logger.info(f"Saved articles with shape {filtered_articles.shape}  to {HM_PATH_PROCESSED}/filtered_articles.csv")

    return filtered_articles


def extract_attrs():
    """
    Preprocess the descriptions of the train set articles
    * keep only nouns and adjectives
    * save the preprocessed descriptions to a new column
    """
    HM_TRAIN_ARTICLES = os.path.join(HM_PATH_PROCESSED, "hm_train.csv")
    # if file dont exists run code
    if os.path.exists(HM_TRAIN_ARTICLES):
        train_df = pd.read_csv(HM_TRAIN_ARTICLES, dtype={"text": "string"})  # to make sure descriptions are strings for processing
        print(f"Loaded train set articles from {HM_TRAIN_ARTICLES} with shape {train_df.shape}")  # should be (104232, 28)
    else:
        print("Creating split..")
        train_df, val_df, test_df = create_split(save_csv=True)

    start_time = time.time()
    train_df = add_noun_adj_column(train_df)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", round(execution_time), "seconds")

    train_df["noun_adj_words"] = train_df["noun_adj_words"].astype(str)  # convert to string
    train_df["noun_adj_words"] = train_df["noun_adj_words"].apply(ast.literal_eval)  # convert to list

    train_df.to_csv(os.path.join(HM_PATH_PROCESSED, "train_with_attrs.csv"), index=False)

    return train_df


def get_attrs_count(df: pd.DataFrame):
    """
    Get the count of each noun and adjective in the DataFrame. For plotting.
    """
    all_words = [word for sublist in df["noun_adj_words"] for word in sublist]
    word_counts = Counter(all_words)
    word_counts_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"])
    return word_counts_df


def add_attrs_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column to the DataFrame containing the extracted lemmatized nouns and adjectives
    """
    nlp = stanza.Pipeline("en", verbose=False)

    logger.info("Processing DataFrame to extract attributes (SINGLE)..")
    
    def extract_noun_adj(sentence):
        try:
            sentence = re.sub(r"-", " ", sentence)  # to merge words that are separated by hyphen

            doc = nlp(sentence)
            noun_adj_words = [
                word.lemma for sent in doc.sentences for word in sent.words if word.upos in ["NOUN", "ADJ", "PROPN"]
            ]  # Extract NOUN, ADJ, and PROPN words from the tokens and lemmatize them
            return noun_adj_words
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Original sentence: {sentence}")
            return []

    df["text"] = df["text"].astype(str)

    tqdm.pandas()
    df["attributes"] = df["text"].progress_apply(extract_noun_adj)
    return df


def add_attrs_column_batch(df: pd.DataFrame, batch_size:int = 500) -> pd.DataFrame:
    """
    Add a new column to the DataFrame containing the extracted lemmatized nouns and adjectives in batches.
    """
    nlp = stanza.Pipeline("en", verbose=False)

    logger.info(f"Processing DataFrame to extract attributes (BATCH={batch_size})..")
    
    def extract_noun_adj_batch(sentences):
        # Process in batch
        docs = nlp(sentences)
        # Extract nouns and adjectives
        results = []
        for doc in docs:
            noun_adj_words = [word.lemma for sent in doc.sentences for word in sent.words if word.upos in ["NOUN", "ADJ", "PROPN"]]
            results.append(noun_adj_words)
        return results

    # Process DataFrame
    sentences = df["text"].astype(str)
    sentences = [re.sub(r"-", " ", sentence) for sentence in sentences]
    in_docs = [stanza.Document([], text=d) for d in sentences]
    tqdm.pandas()
    batches = [in_docs[i:i+batch_size] for i in range(0, len(df), batch_size)]

    processed_batches = []
    for batch in tqdm(batches, desc="Processing batches"):
        processed_batches.extend(extract_noun_adj_batch(batch))

    df["attributes"] = processed_batches
    return df


def plot_word_counts(word_counts_df: pd.DataFrame):
    """
    Plot the word counts as a histogram.
    """

    word_counts_df = word_counts_df.sort_values(by="Count", ascending=False)
    top_20_counts = word_counts_df.head(20)
    print("Top 20 words", top_20_counts)

    sns.set_style("whitegrid")
    scaling_factor = 1
    plt.rcParams.update(
        {
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.labelsize": scaling_factor * 15,
            "font.size": scaling_factor * 16,
            "legend.fontsize": scaling_factor * 12,
            "xtick.labelsize": scaling_factor * 15,
            "ytick.labelsize": scaling_factor * 15,
            "figure.titlesize": scaling_factor * 20,
            "axes.titlesize": scaling_factor * 20,
            "figure.dpi": 96,
        }
    )

    # Plotting the histogram
    plt.figure(figsize=(16, 6))
    plt.bar(top_20_counts["Word"], top_20_counts["Count"], color="rebeccapurple")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top 20 Attributes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("hm_attributes_hist.pdf", dpi=600, bbox_inches="tight")
    plt.show()


def plot_word_cloud(df: pd.DataFrame):
    """
    Generate a word cloud from the 'detail_desc' column of the DataFrame.
    """
    text = " ".join(df["detail_desc"].astype(str))

    x, y = np.ogrid[:1920, :1080]

    mask = (x - 500) ** 2 + (y - 500) ** 2 > 400**2
    mask = 255 * mask.astype(int)

    # Generate a word cloud
    wordcloud = WordCloud(
        width=1920,
        height=1080,
        random_state=42,
        max_words=50,
        mask=mask,
        background_color="white",
    ).generate_from_text(text)
    plt.figure(figsize=(19.20, 10.80))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig("hm_wordcloud.pdf", format="pdf", dpi=300, bbox_inches="tight")

    plt.show()


def copy_image(destination_folder: str, img_path: str):
    """
    Copy images from img_path to destination folder
    """
    file_name = os.path.basename(img_path)
    destination_path = os.path.join(destination_folder, file_name)

    # Check if the source image file exists
    if os.path.exists(img_path):
        if not os.path.exists(destination_path):
            shutil.copyfile(img_path, destination_path)
    else:
        # Print a message for missing image and write to the log file
        message = f"Missing image: {os.path.basename(img_path)} at {img_path}\n"
        logger.debug(message)


def save_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save the train, validation, and test splits as csv files to the HM_SPLIT_FOLDER.
    Creates the folder if it does not exist.
    """
    os.makedirs(os.path.join(HM_SPLIT_FOLDER, "train"), exist_ok=True)
    os.makedirs(os.path.join(HM_SPLIT_FOLDER, "validation"), exist_ok=True)
    os.makedirs(os.path.join(HM_SPLIT_FOLDER, "test"), exist_ok=True)

    tqdm.pandas()
    logger.info(f"Processing train data with shape {train_df.shape}")
    destination_folder = os.path.join(HM_SPLIT_FOLDER, "train")
    train_df["img_path"].progress_apply(lambda img_path: copy_image(destination_folder, img_path))
    logger.info(f"Processing validation data with shape {val_df.shape}")
    destination_folder = os.path.join(HM_SPLIT_FOLDER, "validation")
    val_df["img_path"].progress_apply(lambda img_path: copy_image(destination_folder, img_path))
    logger.info(f"Processing test data with shape {test_df.shape}")
    destination_folder = os.path.join(HM_SPLIT_FOLDER, "test")
    test_df["img_path"].progress_apply(lambda img_path: copy_image(destination_folder, img_path))

    train_df[["file_name", "text"]].to_csv(os.path.join(HM_SPLIT_FOLDER, "train", "metadata.csv"), index=False)
    val_df[["file_name", "text"]].to_csv(os.path.join(HM_SPLIT_FOLDER, "validation", "metadata.csv"), index=False)
    test_df[["file_name", "text"]].to_csv(os.path.join(HM_SPLIT_FOLDER, "test", "metadata.csv"), index=False)


def save_split_arrow():
    """
    Save the train, validation, and test splits as arrow files to the HM_SPLIT_ARROW folder.
    Creates the folder if it does not exist.
    """
    OUTFOLDER = os.getenv("HM_SPLIT_ARROW")
    splits = ["test", "train", "validation"]
    split_dict = {}

    if not os.path.exists(HM_SPLIT_FOLDER):
        logger.info(f"Train split does not exist at {os.path.basename(HM_SPLIT_FOLDER)}. Creating it now.")
        create_split(save=True)

    logger.info(f"Loading splits from {HM_SPLIT_FOLDER}")
    for split in splits:
        print(f"Processing {split} from {os.path.basename(HM_SPLIT_FOLDER)} split and saving to {OUTFOLDER}")
        texts = []
        images = []
        filenames = []
        metadata_path = os.path.join(HM_SPLIT_FOLDER, split, "metadata.csv")
        df = pd.read_csv(metadata_path)
        df["image"] = df["file_name"].apply(lambda image: os.path.join(HM_SPLIT_FOLDER, split, image))
        images.extend(df["image"].tolist())
        texts.extend(df["text"].tolist())
        filenames.extend(df["file_name"].tolist())
        # create a Dataset instance from dict
        ds = Dataset.from_dict({"image": images, "text": texts, "filename": filenames})
        # cast the content of image column to PIL.Image
        ds = ds.cast_column("image", Image())
        # create train split
        split_dict[split] = ds

    dataset = DatasetDict(split_dict)
    dataset.save_to_disk(OUTFOLDER)


def create_split(
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    save_csv: bool = False,
    create_datasets: bool = False,
    create_arrow: bool = False,
):
    """
    Create train, validation, and test splits for the dataset.

    Args:
        train_size (float, optional): The proportion of the dataset to include in the training split. Defaults to 0.8.
        validation_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.1.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        save_csv (bool, optional): Whether to save the splits as CSV files. Defaults to False.
        create_datasets (bool, optional): Whether to create datasets from the splits. Defaults to False.
        create_arrow (bool, optional): Whether to create an arrow dataset. Defaults to False.
    """

    HM_FILTERED_ARTICLES = os.getenv("HM_FILTERED_ARTICLES")
    if os.path.exists(HM_FILTERED_ARTICLES):
        articles = pd.read_csv(HM_FILTERED_ARTICLES)
        logger.info(f"Loaded filtered articles from {HM_FILTERED_ARTICLES} with shape {articles.shape}")  # should be (104660, 25)
    else:
        articles = preprocess_articles(save_file=True)
        logger.info(f"Created filtered articles with shape {articles.shape}")

    logger.info(f"Filtered number of items {len(articles)}")

    articles["colour_group_name_simple"] = articles["colour_group_name"].apply(simplify_color)  # simplify color names
    articles["file_name"] = articles["img_path"].apply(lambda img_path: os.path.basename(img_path))
    articles = articles.rename(columns={"detail_desc": "text"})
    articles["text"] = articles["text"].str.lower()
    articles_shuffled = articles.sample(frac=1, random_state=random_state)

    train_df, temp = train_test_split(articles_shuffled, test_size=(1 - train_size), random_state=random_state)
    val_df, test_df = train_test_split(
        temp,
        test_size=test_size / (test_size + validation_size),
        random_state=random_state,
    )

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if save_csv:
        train_df.to_csv(os.path.join(HM_PATH_PROCESSED, f"hm_train.csv"), index=False)
        val_df.to_csv(os.path.join(HM_PATH_PROCESSED, f"hm_val.csv"), index=False)
        test_df.to_csv(os.path.join(HM_PATH_PROCESSED, f"hm_test.csv"), index=False)

    if create_datasets:
        save_split(train_df, val_df, test_df)

    if create_arrow:
        logger.info("Creating arrow dataset..")
        save_split_arrow()

    return train_df, val_df, test_df


def filter_attributes(df: pd.DataFrame, min_occurences: int = 10) -> list[str]:
    """
    Filter attributes based on their occurrence in the DataFrame. Default attributes that occur at least 10 times are kept.
    Returns a set of frequent words.
    """
    all_words = []

    for attributes in df["attributes"]:
        all_words.extend(attributes)

    word_counts = Counter(all_words)

    frequent_words = {word for word, count in word_counts.items() if count >= min_occurences}

    return frequent_words


def generate_attributes(split: str = "test"):
    """
    Generate attributes from the split articles files. Saves different files for later evaluation (mAP). All words are lowercased.
    """
    split_file = os.path.join(HM_PATH_PROCESSED, f"hm_{split}.csv")
    attrs_file = os.path.join(HM_PATH_PROCESSED, f"{split}_attrs.json")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    intermediate_file = os.path.join(out_dir,f"intermediate_{split}.csv")
    
    if not os.path.exists(attrs_file):
        df = pd.read_csv(split_file, encoding="utf-8")           
        df = add_attrs_column_batch(df) if torch.cuda.is_available() else add_attrs_column(df)
        df.to_csv(intermediate_file, index=False)

        if split == "train":
            attrs_pool = list(filter_attributes(df)) # filter by min_occurences = 10
            attrs_pool = set(attrs_pool)
        else:
            with open(os.path.join(HM_PATH_PROCESSED, "hm_attrs_train.json"), "r", encoding="utf-8") as f:
                attrs_pool = json.load(f)
        attrs_dict = {}

        for index, row in df.iterrows():
            file_name = row["file_name"]
            attributes = " ".join([word for word in row["attributes"] if word in attrs_pool])

            if attributes == "":
                print(f"Empty attributes for {file_name}")

            attrs_dict[file_name] = {"caption": attributes}


        with open(attrs_file, "w", encoding="utf-8") as f:
            json.dump(attrs_dict, f, ensure_ascii=False)  # to include all character


        if split == "train":
            with open(os.path.join(HM_PATH_PROCESSED, "hm_attrs_train.json"), "w", encoding="utf-8") as f:
                json.dump(list(attrs_pool), f, ensure_ascii=False)

        logger.info(f"{split.capitalize()} attributes file created.")
    else:
        logger.info(f"{split.capitalize()} attributes file already exists. Skipping creation.")


def get_categories_mapping(save_file: bool = True):
    """
    Returns a mapping of categories to their respective indices. For classification tasks.
    If save_file is True, saves the mapping to a JSON file.
    """
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
    labels_map = {category: index for index, category in enumerate(categories)}

    if save_file:
        file_path = os.path.join(HM_PATH_PROCESSED, "categories_map.json")
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(labels_map, json_file)

    return labels_map


def generate_categories():
    """
    Generate categories from the split articles files. Saves different files for later evaluation (Acc). All words are lowercased.
    """
    SPLITS_PATH = [
        os.path.join(HM_PATH_PROCESSED, "hm_train.csv"),
        os.path.join(HM_PATH_PROCESSED, "hm_val.csv"),
        os.path.join(HM_PATH_PROCESSED, "hm_test.csv"),
    ]

    labels_map = get_categories_mapping()
    try:
        for split in SPLITS_PATH:
            print("Reading from", split)
            df = pd.read_csv(split)
            df.rename(columns={"product_type_name": "category"}, inplace=True)

            labels = [labels_map[label] for label in df["category"]]

            file_name = os.path.basename(split)
            file_name_without_extension = os.path.splitext(file_name)[0]

            output_file_path = os.path.join(HM_PATH_PROCESSED, file_name_without_extension[3:] + "_categories.json")
            with open(output_file_path, "w", encoding="utf-8") as json_file:
                json.dump(labels, json_file)
            print("Categories saved to file:", output_file_path, "\n")
    except FileNotFoundError:
        print("File not found. Please make sure the splits are saved in the correct folder or a generated first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for processing hm data.")
    parser.add_argument(
        "--steps",
        nargs="+",
        help="List of steps to run for preprocessing: split, cates, attrs",
    )
    parser.add_argument(
        "-sc",
        "--save_csv",
        action="store_false",
        help="Save data splits as CSV (default: False)",
    )
    parser.add_argument(
        "-cd",
        "--create_datasets",
        action="store_false",
        help="Create .csv files for dataset splits (default: False)",
    )
    parser.add_argument(
        "-ca",
        "--create_arrow",
        action="store_false",
        help="Create Arrow files for torch.Dataset usage (faster to load in) (default: False)",
    )
    parser.add_argument(
        "--attrs_split",
        type=str,
        default="test",
        help="Split for attributes generation: train, test, or val (default: test)",
    )
    args = parser.parse_args()
    steps = args.steps

    if "split" in steps:
        create_split(
            save_csv=args.save_csv,
            create_datasets=args.create_datasets,
            create_arrow=args.create_arrow,
        )

    if "attrs" in steps:
        split = args.attrs_split.lower()
        assert split in {"train", "val", "test"}
        generate_attributes(split)

    if "cates" in steps:
        generate_categories()