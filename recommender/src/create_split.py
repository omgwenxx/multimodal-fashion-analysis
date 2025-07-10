import polars as pl
import pandas as pd
import os
import argparse
import shutil
import json
import math
from dotenv import load_dotenv

load_dotenv()

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    df = pl.read_csv(os.path.join(os.getenv("HM_RAW"), 'transactions_train.csv'))
    articles = pl.read_csv(os.getenv("HM_FILTERED_ARTICLES"))
    
    print(f"Loaded dataset with {len(df)} samples. Should be 31,788,324.")
    print(f"Loaded filtered articles with {len(articles)} samples. Should be 104,232.")

    df = df.with_columns(pl.lit(1).alias("rating"))
    df = df.with_columns(pl.col("t_dat").str.strptime(pl.Date, "%Y-%m-%d").alias("t_dat"))
    df = df.select(['customer_id', 'article_id', 'rating', 't_dat'])
    df = df.filter(pl.col("article_id").is_in(articles["article_id"]))

    print(f"Filtered dataset with {len(df)} samples. Should be 31,400,864.")
    
    return df

def sample_data(df, max_rows=2_000_000):
    """
    Sample the dataset by taking the first transaction per customer.
    If more rows are needed, iteratively add transactions in chunks.
    """
    # Step 1: Sort transactions by timestamp within each customer group
    df = df.sort(["customer_id", "t_dat"])

    # Step 2: Get the first transaction per customer
    first_transactions = df.group_by("customer_id").head(1)
    sampled_df = first_transactions
    remaining_rows = max_rows - len(sampled_df)

    # Step 3: If we need more rows, iteratively add transactions in chunks
    if remaining_rows > 0:
        df_remaining = df.join(first_transactions, on=["customer_id", "t_dat"], how="anti")

        batch_size = 50_000  # Process transactions in batches to avoid memory overload
        while remaining_rows > 0 and len(df_remaining) > 0:
            extra_transactions = df_remaining.group_by("customer_id").head(1)  # Take 1 extra transaction per customer

            # Limit batch size to avoid excessive memory usage
            extra_transactions = extra_transactions.head(min(batch_size, remaining_rows))
            
            sampled_df = sampled_df.vstack(extra_transactions)
            remaining_rows -= len(extra_transactions)

            # Remove sampled transactions from remaining data
            df_remaining = df_remaining.join(extra_transactions, on=["customer_id", "t_dat"], how="anti")

    return sampled_df

def sample_dataset_by_mode(df, mode):
    """Split the dataset based on the selected mode."""
    if mode == "test":
        df = sample_data(df)
    elif mode == "simple":
        df = sample_data(df,10_000_000)
    elif mode == "full":
        pass  # Use full dataset
    else:
        raise ValueError("Invalid mode. Choose from 'test', 'simple', or 'full'.")
    
    return df

def create_split(data, ratio=0.2):
    tuple_list = []
    data = data.to_pandas()
    user_size = data.groupby(['customer_id'], as_index=True).size()
    user_threshold = user_size.apply(lambda x: math.floor(x * (1 - ratio)))
    data['rank_first'] = data.groupby(['customer_id'])['t_dat'].rank(method='first', ascending=True)
    data["test_flag"] = data.apply(
        lambda x: x["rank_first"] > user_threshold.loc[x["customer_id"]], axis=1)
    test = data[data["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    train = data[data["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    tuple_list.append((train, test))

    print(f"Train: {len(train)} samples, Test: {len(test)} samples")
    return tuple_list

def create_folder_by_index(path, index):
    """
    Code from elliot.
    """
    if os.path.exists(os.path.abspath(os.sep.join([path, index]))):
        shutil.rmtree(os.path.abspath(os.sep.join([path, index])))
    os.makedirs(os.path.abspath(os.sep.join([path, index])))
    return os.path.abspath(os.sep.join([path, index]))

def store_splitting(save_folder, tuple_list):
    """
    Code from elliot.
    """
    for i, (train_val, test) in enumerate(tuple_list):
        actual_test_folder = create_folder_by_index(save_folder, str(i))
        test.to_csv(os.path.abspath(os.sep.join([actual_test_folder, "test.tsv"])), sep='\t', index=False, header=False)
        if isinstance(train_val, list):
            for j, (train, val) in enumerate(train_val):
                actual_val_folder = create_folder_by_index(actual_test_folder, str(j))
                val.to_csv(os.path.abspath(os.sep.join([actual_val_folder, "val.tsv"])), sep='\t', index=False, header=False)
                train.to_csv(os.path.abspath(os.sep.join([actual_val_folder, "train.tsv"])), sep='\t', index=False, header=False)
        else:
            train_val.to_csv(os.path.abspath(os.sep.join([actual_test_folder, "train.tsv"])), sep='\t', index=False, header=False)

def index_data(output_path):
    """Read user and item maps and apply indexing to train and test data."""
    with open(os.getenv("HM_USERS_MAP")) as f:
        users_map = json.load(f)
    with open(os.getenv("HM_ITEMS_MAP")) as f:
        items_map = json.load(f)
    
    train = pd.read_csv(os.path.join(output_path,'train.tsv'), sep='\t', header=None)
    test = pd.read_csv(os.path.join(output_path,'test.tsv'), sep='\t', header=None)
    
    train[0] = train[0].map(users_map)
    train[1] = train[1].astype(str).map(items_map)
    test[0] = test[0].map(users_map)
    test[1] = test[1].astype(str).map(items_map)
    
    train.to_csv(os.path.join(output_path,'train_indexed.tsv'), sep='\t', index=False, header=None)
    test.to_csv(os.path.join(output_path,'test_indexed.tsv'), sep='\t', index=False, header=None)


def main():
    parser = argparse.ArgumentParser(description="Dataset Splitter for H&M Fashion Recommendations")
    parser.add_argument("mode", choices=["test", "simple", "full"], help="Dataset split mode")
    args = parser.parse_args()
    
    df = load_and_preprocess_data()
    sample = sample_dataset_by_mode(df, args.mode)
    tuple_list = create_split(sample)
    output_path = os.path.join(os.getenv("HM_PROCESSED"), f"hm_rec_splits_{args.mode}")
    store_splitting(output_path, tuple_list)
    index_data(output_path)
    print(f"Dataset saved to {output_path} with {len(sample)} samples.")

if __name__ == "__main__":
    main()
