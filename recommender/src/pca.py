import os
import glob
import numpy as np
from sklearn.decomposition import IncrementalPCA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
input_folder = f"{os.getenv("HM_RAW")}/blip2_features/visual_features"  # Folder containing .npy feature vectors
output_folder = f"{os.getenv("HM_RAW")}/blip2_features/visual_features_reduced_pca"
os.makedirs(output_folder, exist_ok=True)

batch_size = 1000  # Process data in batches

# --- Load file paths ---
file_paths = glob.glob(os.path.join(input_folder, "*.npy"))
file_paths.sort()  # Sort files alphabetically

article_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]

# Load one sample to determine feature size
sample_vector = np.load(file_paths[0])
feature_size = sample_vector.shape[0]
num_samples = len(file_paths)

print(f"Detected {num_samples} samples, each with {feature_size} features.")

# --- First Pass: Fit on a Small Subset to Determine n_components ---
ipca_temp = IncrementalPCA(n_components=min(200, feature_size), batch_size=batch_size)  # Initial estimate
subset_size = min(5000, num_samples)  # Use a smaller subset if dataset is huge
subset_vectors = np.array([np.load(f) for f in file_paths[:subset_size]])

ipca_temp.fit(subset_vectors)

# Find how many components retain 95% variance
explained_variance = np.cumsum(ipca_temp.explained_variance_ratio_)
n_components = np.searchsorted(explained_variance, 0.95) + 1

print(f"Selected n_components = {n_components} (to retain 95% variance)")

# --- Second Pass: Apply Incremental PCA with Determined n_components ---
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

for start in range(0, num_samples, batch_size):
    batch_files = file_paths[start:start + batch_size]
    batch_vectors = np.array([np.load(f) for f in batch_files])
    ipca.partial_fit(batch_vectors)
    print(f"Processed batch {start // batch_size + 1} / {num_samples // batch_size + 1}")

print("PCA fitting complete.")

# --- Third Pass: Transform and Save ---
for start in range(0, num_samples, batch_size):
    batch_files = file_paths[start:start + batch_size]
    batch_vectors = np.array([np.load(f) for f in batch_files])
    batch_reduced = ipca.transform(batch_vectors)

    for article_id, reduced_vector in zip(article_ids[start:start + batch_size], batch_reduced):
        output_filename = f"{article_id}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, reduced_vector)

    print(f"Saved batch {start // batch_size + 1} / {num_samples // batch_size + 1}")

print(f"Reduced feature vectors saved in folder: {output_folder}")

# --- Print shape of the first reduced vector ---
print("Shape of the first reduced vector:", batch_reduced[0].shape)
