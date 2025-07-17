<!-- TOC --><a name="hm-data-preprocessing"></a>
# HM Data Preprocessing

Sure! Here's the full text version, ready to copy into your Markdown file:

---

## 🛠 How to Run `preprocess_hm.py`

This script is used to preprocess H\&M data by running one or more preprocessing steps such as splitting the data, generating categories, or extracting attributes.

### 📄 Usage

```
python preprocess_hm.py --steps [STEP ...] [OPTIONS]
```

### 🧩 Available Steps

Pass one or more of the following values to `--steps` to define which parts of the pipeline to run:

* `split` – create data splits
* `attrs` – generate attribute labels
* `cates` – generate category labels

You can run multiple steps at once:

```
python preprocess_hm.py --steps split attrs cates
```

### ⚙️ Optional Flags

* `--save_csv` or `-sc`: Save split data as CSV (default: not saved unless this flag is set)
* `--create_datasets` or `-cd`: Create .csv files for dataset splits (default: not created unless flag is set)
* `--create_arrow` or `-ca`: Save Arrow files for faster loading (default: not created unless flag is set)
* `--attrs_split SPLIT`: Choose the split for generating attributes (`train`, `val`, `test`; default: `test`)

### ✅ Example Commands

Generate all splits and datasets:

```
python preprocess_hm.py --steps split --save_csv --create_datasets --create_arrow
```

Generate only attribute labels for validation split:

```
python preprocess_hm.py --steps attrs --attrs_split val
```

Generate categories only:

```
python preprocess_hm.py --steps cates
```


## Preprocessing pipeline
<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->
More information on the preprocessing steps, can be found here.
   * [Process](#process)
      + [Setup](#setup)
      + [Preprocessing](#preprocessing)
         - [Remove items with missing description or image](#remove-items-with-missing-description-or-image)
         - [Filter items by category](#filter-items-by-category)
         - [Process descriptions and attribute extraction](#process-descriptions-and-attribute-extraction)
         - [Create split](#create-split)
      + [Commandline Tool](#commandline-tool)
- [Split and preprocessing numbers](#split-and-preprocessing-numbers)
- [Sped up data loading using arrow](#sped-up-data-loading-using-arrow)

<!-- TOC end -->


<!-- TOC --><a name="process"></a>
## Process
<!-- TOC --><a name="setup"></a>
### Setup
We load data that was downloaded from [kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations). The folder structure is as follows:
```
h-and-m-personalized-fashion-recommendations/
├── images/
├── articles.csv
├── customers.csv
├── sample_submission.csv
└── transactions_train.csv
```

We only need articles and images to create the image captioning dataset.

<!-- TOC --><a name="preprocessing"></a>
### Preprocessing
<!-- TOC --><a name="remove-items-with-missing-description-or-image"></a>
#### Remove items with missing description or image
Running the following code removes items without a description (`detail_desc` column) or without a corresponding image.
```python
from tqdm import tqdm
import pandas as pd
import os

HM_PATH = config["Folders"]["hm_raw"]
articles = pd.read_csv(f"{HM_PATH}/articles.csv")

IMAGES_PATH = config["Folders"]["hm_images"]

# remove missing description or missing image
articles = articles.dropna(subset=['detail_desc']) # remove rows with missing descriptions
def create_img_folder(article_id):
    return f'0{str(article_id)[:2]}'

def create_img_id(article_id):
    return f'0{str(article_id)}'

articles['img_folder'] = articles['article_id'].apply(create_img_folder)
articles['img_id'] = articles['article_id'].apply(create_img_id)
articles['img_path'] = IMAGES_PATH + "/" + articles['img_folder'] + '/' + articles['img_id'] + ".jpg"

# Remove items without image
def image_exists(img_path):
    return os.path.exists(img_path)

tqdm.pandas()
articles = articles[articles['img_path'].progress_apply(image_exists)]
```

This reduces our initial number from 105542 to 104696.

<!-- TOC --><a name="filter-items-by-category"></a>
#### Filter items by category
Taking a look at the dataframe, we see that we have 3 hierarchical product categories `product group` -> `product type` -> `product name`. We will investigate the granularity in the next step.
```python
unique_product_groups = articles['product_group_name'].unique()
print('Unique values in the product_group_name column:')
print(len(unique_product_groups))

unique_product_type_names = articles['product_type_name'].unique()
print('Unique values in the product_type_name column:')
print(len(unique_product_type_names))

unique_product_names = articles['prod_name'].unique()
print('Unique values in the prod_name column:')
print(len(unique_product_names))
```
Output:
```
Unique values in the product_group_name column:
19
Unique values in the product_type_name column:
131
Unique values in the prod_name column:
45567
```

Product type seems to be the reasonable choice as product group is too general and single product names are too specific. We now analyze how many items are there per product type.

```python
import numpy as np

def quartiles(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    return q1, q2, q3

q1, q2, q3 = quartiles(category_counts)
print("Q1:", q1)
print("Q2 (Median):", q2)
print("Q3:", q3)
```
Output:
```python
Q1: 7.0
Q2 (Median): 113.0
Q3: 686.0
```

We have 25% of the item categories having less than 7 items, therefor we decide to keep only categories that have 7 or more items. This reduces the number of categories from 131 to 100.
```python
category_counts = articles['product_type_name'].value_counts()

filtered_categories = category_counts[category_counts >= 7]

print('Categories with at least 7 items:')
filtered_categories.index
len(filtered_categories.index)
```
Output:
```python
Index(['Trousers', 'Dress', 'Sweater', 'T-shirt', 'Top', 'Blouse', 'Jacket',
       'Shorts', 'Shirt', 'Vest top', 'Underwear bottom', 'Skirt', 'Hoodie',
       'Bra', 'Socks', 'Leggings/Tights', 'Sneakers', 'Cardigan', 'Hat/beanie',
       'Garment Set', 'Swimwear bottom', 'Bag', 'Earring', 'Jumpsuit/Playsuit',
       'Pyjama set', 'Blazer', 'Boots', 'Other accessories', 'Scarf',
       'Bodysuit', 'Bikini top', 'Hair/alice band', 'Sandals', 'Swimsuit',
       'Cap/peaked', 'Sunglasses', 'Necklace', 'Underwear Tights', 'Coat',
       'Polo shirt', 'Belt', 'Hat/brim', 'Pyjama jumpsuit/playsuit',
       'Other shoe', 'Gloves', 'Ballerinas', 'Dungarees', 'Slippers',
       'Hair clip', 'Ring', 'Hair string', 'Pyjama bottom', 'Heeled sandals',
       'Swimwear set', 'Pumps', 'Night gown', 'Underwear body', 'Bracelet',
       'Flat shoe', 'Outdoor Waistcoat', 'Tie', 'Robe', 'Outdoor trousers',
       'Flip flop', 'Unknown', 'Wedge', 'Kids Underwear top', 'Costumes',
       'Wallet', 'Watch', 'Tailored Waistcoat', 'Sarong', 'Outdoor overall',
       'Beanie', 'Swimwear top', 'Sleeping sack', 'Underwear set',
       'Fine cosmetics', 'Soft Toys', 'Bootie', 'Long John', 'Umbrella',
       'Hair ties', 'Waterbottle', 'Heels', 'Dog Wear', 'Underdress',
       'Nipple covers', 'Giftbox', 'Side table', 'Cap', 'Earrings', 'Felt hat',
       'Flat shoes', 'Weekend/Gym bag', 'Leg warmers', 'Dog wear',
       'Bucket hat', 'Underwear corset', 'Accessories set'],
      dtype='object', name='product_type_name')
100
```

We still have a lot of fashion unrelated categories e.g. Dog wear or Sleepin Sack, we therefor manually removed all unrelated categories.


```python
def filter_fashion_and_accessory_related(product_types):
    fashion_keywords = {'Wedge','Trousers', 'Dress', 'Sweater', 'T-shirt', 'Top', 'Blouse', 'Jacket', 'Shorts', 'Shirt', 'Vest top', 'Underwear bottom', 'Skirt', 'Hoodie', 'Bra', 'Socks', 'Leggings/Tights', 'Sneakers', 'Cardigan', 'Garment Set', 'Swimwear bottom', 'Bag', 'Jumpsuit/Playsuit', 'Pyjama set', 'Blazer', 'Bodysuit', 'Bikini top', 'Sandals', 'Swimsuit', 'Sunglasses', 'Underwear Tights', 'Coat', 'Polo shirt', 'Pyjama jumpsuit/playsuit', 'Ballerinas', 'Dungarees', 'Slippers', 'Pyjama bottom', 'Heeled sandals', 'Swimwear set', 'Pumps', 'Underwear body', 'Night gown', 'Flat shoe', 'Outdoor Waistcoat', 'Robe', 'Outdoor trousers', 'Flip flop', 'Kids Underwear top', 'Costumes', 'Tailored Waistcoat', 'Sarong', 'Outdoor overall', 'Swimwear top', 'Bootie', 'Long John', 'Hair ties', 'Heels', 'Underdress', 'Flat shoes', 'Leg warmers', 'Bucket hat', 'Underwear corset', 'Boots', 'Tie', 'Cap', 'Other accessories', 'Hat/brim', 'Accessories set', 'Other shoe', 'Underwear set'}
    accessory_keywords = {'Earring', 'Hat/beanie', 'Necklace', 'Hair/alice band', 'Hair clip', 'Ring', 'Hair string', 'Bracelet', 'Scarf', 'Cap/peaked', 'Belt', 'Gloves', 'Wallet', 'Watch', 'Beanie', 'Heels', 'Earrings', 'Felt hat', 'Weekend/Gym bag'}

    filtered_product_types = set()
    for ptype in product_types:
        if any(keyword.lower() in ptype.lower() for keyword in fashion_keywords) or any(keyword.lower() in ptype.lower() for keyword in accessory_keywords):
            filtered_product_types.add(ptype)
    return list(filtered_product_types)

filtered_product_types = filter_fashion_and_accessory_related(filtered_categories.index)
print(len(filtered_product_types))
```
Output:
```python
89
```

The final list now inluces only fashion related categories, this also includes shoes, bags and accessoirs. We now can filter the initial dataframe to only keep articles that are within those categories and have a description. A list of all filtered categories can be found in the `files/not_included_categories_all.txt`.

<!-- TOC --><a name="process-descriptions-and-attribute-extraction"></a>
#### Process descriptions and attribute extraction
To compute the mean average precision, we need to extract the attributes in the descriptions. For this we use [stanza](https://stanfordnlp.github.io/stanza/) Version 1.8.2, to extract nouns, adjectives and proper nouns according to [Universal POS definitions](https://universaldependencies.org/u/pos/).

For this step we use the prefiltered articles (only items with image and caption, categories with at least 7 items, only fashion and accesory related) with 104232 articles.

```python
import re
from collections import Counter
import stanza
import ast
stanza.download('en') # download English model
nlp = stanza.Pipeline('en')import stanza

# Function to apply pos_tag and extract NOUN, ADJ and PROPN words
def extract_noun_adj(sentence):
    try:
        # Tokenize the sentence
        sentence = re.sub(r'-', '', sentence) # to not split T-shirt
        doc = nlp(sentence)
        # Extract NOUN, ADJ, and PROPN words from the tokens
        noun_adj_words = [word.text for sent in doc.sentences for word in sent.words
                          if word.upos in ['NOUN', 'ADJ', 'PROPN']]
        return noun_adj_words
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Original sentence: {sentence}")
        return []

start_time = time.time()

df['detail_desc'] = df['detail_desc'].astype(str) # needs to be str

tqdm.pandas()
df['noun_adj_words'] = df['detail_desc'].progress_apply(extract_noun_adj)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
# This can take up to 5h

# Flatten the lists of words
df['noun_adj_words'] = df['noun_adj_words'].astype(str)
df['noun_adj_words'] = df['noun_adj_words'].apply(ast.literal_eval)
all_words = [word for sublist in df['noun_adj_words'] for word in sublist]

# Count the occurrences of each word
word_counts = Counter(all_words)

# Create a new DataFrame from the word counts
word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
print(len(all_words)) # 1360122
```
After this, we saved lists of attributes to the column `noun_adj_words`. We extract a total of 1360122 attributes. However, we only want to keep meaningful ones, therefor we only keep attributes that appear at least 10 times. 

```python
# Sort the DataFrame by 'Count' column in descending order
word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

# Display the top 20 count values
top_20_counts = word_counts_df.head(20)
print(top_20_counts)
```
Output:
```python
            Word  Count
90         front  35790
16          back  34137
43         waist  34018
49        cotton  33214
47          soft  30234
61       sleeves  28327
1            top  26129
55        jersey  25798
167          hem  25029
67       pockets  24837
159        cuffs  23632
64        fabric  18071
65   elasticated  16375
60          long  14654
56          side  13887
245        weave  13857
123          zip  13117
59      neckline  12783
318       button  11738
164       collar  11243
```

```python
# Step 3: Identify words that appear in at least 10 items
words_to_keep = {word for word, count in word_counts.items() if count >= 10}

# Step 4: Filter each list in 'noun_adj_words' to only include these words
df['filtered_noun_adj_words'] = df['noun_adj_words'].apply(lambda x: [word for word in x if word in words_to_keep])

len(words_to_keep) #1775
```
We end up with 1775 attributes in total. An example of an item with a filtered list is shown below.

<!-- TOC --><a name="create-split"></a>
#### Create split
We then create a train-validation-test split with the ratios 8-1-1. For reproducability, we set a random seed to 42. Assuming the `articles` dataframe represents the final dataframe after the previous two step, we can reproduce the split with the following code:
```python
    # Create train, val, test split
    train_size = 0.8
    validation_size = 0.1
    test_size = 0.1 
    random_state = 42
    selected_columns = ['img_path','img_id','detail_desc','product_type_name','noun_adj_words',"colour_group_name",'filtered_noun_adj_words']
    articles_imgcap = articles[selected_columns].copy()

    articles_imgcap['filtered_noun_adj_words'] = articles_imgcap['filtered_noun_adj_words'].apply(ast.literal_eval) # convert to list
    articles_imgcap['colour_group_name_simple'] = articles_imgcap['colour_group_name'].apply(simplify_color) # simplify color names
    articles_imgcap['attributes'] = articles_imgcap.apply(lambda row: [row['colour_group_name_simple']] + row['filtered_noun_adj_words'] if row['colour_group_name_simple'] != 'other' else row['filtered_noun_adj_words'], axis=1) # combine color and attributes

    articles_imgcap["file_name"] = articles_imgcap["img_path"].apply(lambda img_path:os.path.basename(img_path))
    articles_imgcap = articles_imgcap.rename(columns={"detail_desc":"text"})
    articles_shuffled = articles_imgcap.sample(frac=1, random_state=42)

    train_df, temp = train_test_split(articles_shuffled, test_size=(1 - train_size), random_state=random_state)
    val_df, test_df = train_test_split(temp, test_size=test_size/(test_size + validation_size), random_state=random_state)

    # Optionally, reset the index of the resulting dataframes
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv("hm_train.csv"), index=False)
    val_df.to_csv("hm_val.csv"), index=False)
    test_df.to_csv("hm_test.csv"), index=False)
```

<!-- TOC --><a name="commandline-tool"></a>
### Commandline Tool
You can also just call the scrip `preprocess_hm.py` from the command line and all train, validation and test files and folders are going to be created.

#### Command-line Arguments

- `--steps`: List of steps to run for preprocessing. Possible values: `split`, `cates`, `attrs`.

**Steps**

**Split**\
Creates train, validation and test split. Saves files and images in folders according to their split, so that they can
be loaded in via huggingface [image folder tutorial](https://huggingface.co/docs/datasets/image_dataset).
- `-sc`, `--save_csv`: Save data splits as CSV (default: False).
- `-cd`, `--create_datasets`: Create datasets for torch.Dataset usage (default: False).
- `-ca`, `--create_arrow`: Create Arrow files for torch.Dataset usage (faster to load in) (default: False).

```bash
python -m preprocessing.preprocess_hm --steps split --save_csv --create_datasets --create_arrow
```

Generate attributes for the dataset.
```bash
python -m preprocessing.preprocess_hm --steps attrs
```

Generate categories for the dataset. Create a file for each split with a list of numbers representing the categories.
Mapping will be save to a file called `categories_map.json` with a dictionary of the mappings.
```bash
python -m preprocessing.preprocess_hm --steps cates
```

Example
```bash
# Run split step, save data splits as CSV, create datasets, and create Arrow files
python -m preprocessing.preprocess_hm --steps split --save_csv --create_datasets --create_arrow

# Run attributes step
python -m preprocessing.preprocess_hm --steps attrs

# Run categories step
python -m preprocessing.preprocess_hm --steps cates
```

<!-- TOC --><a name="split-and-preprocessing-numbers"></a>
# Split and preprocessing numbers
Initial number of items - 105542\
After removing items without description - 105126\
After removing items without corresponding image - 104696\
After keeping items of categories with at least 7 items - 104232

Preprocessing steps:
* Remove items without image or detail_desc
* Remove items of categories that do not have at least 7 items
* Remove items of non-fashion or non-accessory related categories (see not_included_categories.txt)

Final number of articles 104232

Train - 83385 (80%)\
Validation - 10423 (10%)\
Test - 10424(10%)

<!-- TOC --><a name="sped-up-data-loading-using-arrow"></a>
# Sped up data loading using arrow
https://ngwaifoong92.medium.com/how-to-speed-up-data-loading-for-machine-learning-training-d10d202d5d13
```python
import os
import json
from datasets import DatasetDict, Dataset, Image
import pandas as pd

data_folder = "./train_split"
outfolder = "./train_split_arrow"
splits = ["test","train","validation"]

split_dict = {}

for split in splits:
    print(f"Processing {split} split")
    texts = []
    images = []
    filenames = []
    metadata_path = os.path.join(data_folder, split, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df["image"] = df["file_name"].apply(lambda image: os.path.join(data_folder, split, image))
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
dataset.save_to_disk(outfolder)
```

Then load with
```python
dataset = load_from_disk("./train_split_arrow/train") # specifying split
dataset = load_from_disk("./train_split_arrow") # load all splits
```
