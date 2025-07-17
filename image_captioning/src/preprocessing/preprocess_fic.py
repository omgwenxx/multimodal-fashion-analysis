import logging
import json
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv


logger = logging.getLogger("preprocessing.preprocess_fic")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
fic_dataset = os.getenv("FACAD")
word_map_path = os.getenv("FIC_WORDMAP")

data_folder = fic_dataset
meta_file = "meta_all_129927.json"
captions_file = "TEST_CAPTIONS_RAW.json"
test_cap_attrs_map = "TEST_CAP_TO_ATTRS_MAP.json"
    
def map_captions_to_attrs(output_file: str = "TEST_CAP_TO_ATTRS_MAP.json"):
    """
    Maps unique captions from a JSON file to attributes based on metadata and saves the result to a JSON file.
    This is done because attributes file for the test set does not match the test items.
    
    Parameters:
        output_file (str): Filename for the output JSON file.
    """

    def calculate_overlap(main_string, substring):
        """
        Calculate the percentage of overlap between two strings based on word tokens.
        :param main_string: The main string to compare against.
        :param substring: The substring to check for overlap.
        :return: Percentage of overlap between the two strings.
        """
        sub_len = len(substring.lower().split())
        main_words = set(main_string.lower().split()[:sub_len])
        sub_words = set(substring.lower().split())
        
        overlap = main_words.intersection(sub_words)
        overlap_percentage = len(overlap) / len(sub_words)
        
        return overlap_percentage

    with open(os.path.join(data_folder, meta_file), 'r') as file:
        meta_data = json.load(file)

    with open(os.path.join(data_folder, captions_file), 'r') as file:
        test_caps = json.load(file)

    unique_captions_dict = {caption: None for caption in test_caps}

    for caption in tqdm(unique_captions_dict.keys(), desc="Processing captions"):
        best_match = None
        highest_score = 0
        for item in meta_data:
            overlap_score = calculate_overlap(item['description'], caption)
            if overlap_score > highest_score:
                best_match = item['attr']
                highest_score = overlap_score
        unique_captions_dict[caption] = best_match

    output_json_path = os.path.join(data_folder, output_file)
    with open(output_json_path, 'w') as json_file:
        json.dump(unique_captions_dict, json_file, indent=4)

    print(f"JSON file has been saved to {output_json_path}")


def decode_word_ids(file_path):
    """
    Decodes a whole file of encoded captions to a file with the captions in natural language removing nah words.
    The final file has a list of dict where each key represent an image id and the values are dicts with the key
    being "caption" and the value being caption in natural language.
    """
    file_name = os.path.splitext(file_path)[0]
    with open(file_path, "r") as j:
        file = json.load(j)

    decoded_captions = {}
    for idx, caption in enumerate(tqdm(file, desc="Decoding captions")):
        decoded_caption = tokens_to_caption(caption)
        decoded_captions[idx] = {"caption": decoded_caption}

    with open(os.path.join(fic_dataset, file_name + "_RAW.json"), "w") as f:
        json.dump(decoded_captions, f)


def tokens_to_caption(caption):
    """
    Transform tokens to a sentence and removes nah words (relevant for attributes files)
    Using the word map file from the facad dataset
    :param tokens: tokens from dataset
    :return: string with sentence
    """
    with open(word_map_path, "r") as f:
        word_map = json.load(f)

    dictionary = {x: y for y, x in word_map.items()}
    img_caption = [dictionary[w] for w in caption if w not in [0, 15804, 15805, 15806] and dictionary[w] != "nah"]
    cap = " ".join(img_caption)
    return cap
