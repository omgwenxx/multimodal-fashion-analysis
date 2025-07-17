# Comparative Analysis of Fashion Captioning and Multimodal Fashion Recommendation
## Image captioning
### Code
You find all code related to image captioning in the `image_captioning/src` folder.

The following folders (and corresponding code) are included:
- `app` folder including code to run the streamlit app for qualitative evaluation and a file with the first 50 items for each dataset and the corresponding results from each model.
- `models` folder includes the python code for the model wrappers. They load in the models in a unified format for later use. The hugging face models were previously downloaded into a separate folder to avoid downloading everytime a model object was created.
- `preprocessing` folder includes all the code used for preprocessing the datasets. As well as an additional README with more information.
- `pyevalcap` includes code from Chen et al. ([Github repo](https://github.com/salaniz/pycocoevalcap)) which was adjusted to our setup. As well as to additional metric modules map and accuracy.
- `tasks` folder includes all code that runs a full pipeline like training, evaluation and inference. `train.py` includes the code for training BLIP2 and `train_llava.py` for LLaVa.

---
### Preprocessing H&M
To preprocess the item descriptions to extract the attributes of items (based on [Yang et al.](https://arxiv.org/pdf/2008.02693.pdf)), we used [Stanza](https://stanfordnlp.github.io/stanza/) an updated version from the original Stanford Parser. More information on the preprocessing can be found in the README.md of the `image_captioning/src/preprocessing` folder.

---
### Evaluation
To run the evaluation you first need to run
```
chmod +x get_stanford_models.sh 
./get_stanford_models.sh
```
which will unpack the necessary Stanford CoreNLP packages into the spice package.

Missing paraphrase file for the meteor package can be found [here](https://github.com/tylin/coco-caption/blob/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/meteor/data/paraphrase-en.gz).

Here’s a clean, copy-paste-ready **"Training"** section for your `README.md`, based on your fine-tuning and hyperparameter tuning setup:

---

### Training

#### Fine-Tuning with LoRA and rsLoRA

We fine-tune vision-language models using **Low-Rank Adaptation (LoRA)** and the enhanced **Rank-Stabilized LoRA (rsLoRA)**. This approach significantly reduces the number of trainable parameters by decomposing weight updates into two smaller matrices, enabling efficient fine-tuning even on large-scale models.

We use the following LoRA parameters:

* **Rank**: `32`
* **Alpha**: `2 × rank = 64`
* **Dropout**: `0.05`
* **LoRA Layers**: either `all-linear` (QLoRA-style) or `query+value` (default LoRA)
* **rsLoRA Scaling**: `alpha / sqrt(rank)` for improved stability

All models were fine-tuned on A100 80GB GPUs using `bfloat16` precision to prevent numerical instability and reduce memory usage.

#### Hyperparameter Tuning

We ran a grid search to tune:

* **Learning Rate**: `{1e-5, 5e-5, 1e-4, 5e-4}`
* **Effective Batch Size**: `{16, 32}` (with gradient accumulation if needed)
* **LoRA Layers**: `{all-linear, query+value}`

We tested 16 combinations on the smallest model (BLIP-2 2.7B) and the H\&M dataset to identify the best setup.

**Best configuration selected** (based on validation loss and efficiency):

* Learning rate: `1e-4`
* Batch size: `16`
* LoRA layers: `query+value`

This configuration offered a good trade-off between convergence time, stability, and resource efficiency.

#### Environmental Impact

Training was conducted on a university GPU cluster (A100 SXM4 80GB), where one full training run (168h) emitted an estimated **29.03 kg CO₂**. To reduce environmental impact, the hyperparameter sweep was limited to one model and one dataset.

#### Learning Rate Scheduling and Optimization

* **Scheduler**: `CosineAnnealingLR`
* **Initial LR**: `5e-4`, with min LR of `1e-6`, `T_max=500`
* **Weight Decay**: `1e-6`

The cosine schedule was chosen to allow large updates early in training and smaller updates later for fine convergence.

---

### Results
Here are the results with **pre-trained (PT)** and **fine-tuned (FT)** results shown in the format `PT / FT` side by side in each cell.

---

### 🧥 H\&M Fashion Captioning Results (PT/FT)

| **Model**       | **BLEU-4**     | **METEOR**     | **ROUGE-L**     | **CIDEr**\*     | **SPICE**      | **MAP / MAR**                 | **Acc**         |
| --------------- | -------------- | -------------- | --------------- | --------------- | -------------- | ----------------------------- | --------------- |
| BLIP-2-2.7B     | 0.3 / 40.8     | 5.3 / 33.5     | 14.8 / 63.4     | 7.0 / 275.2     | 8.0 / 44.7     | 37.2 / 11.4 → 70.8 / 65.4     | 53.0 / 83.5     |
| **BLIP-2-6.7B** | 0.3 / **41.4** | 5.5 / **33.9** | 15.3 / **63.8** | 7.3 / **281.3** | 8.3 / **45.4** | 38.3 / 11.6 → **70.8 / 66.0** | 53.8 / **83.8** |
| BLIP-2-XL       | 0.3 / 0.2      | 5.1 / 4.4      | 15.0 / 10.6     | 6.3 / 6.6       | 8.0 / 7.3      | 38.9 / 10.8 → 37.0 / 10.3     | 56.5 / 51.7     |
| BLIP-2-XXL      | 0.3 / 0.4      | 5.3 / 5.6      | 15.5 / 16.6     | 7.0 / 7.2       | 8.4 / 8.7      | 38.0 / 11.2 → 38.2 / 11.9     | 58.0 / 56.0     |
| LLaVA-1.5-7B    | 0.5 / 23.4     | 7.4 / 32.8     | 14.5 / 46.0     | 0.8 / 24.2      | 5.3 / 34.6     | 15.2 / 13.0 → 53.1 / **71.3** | 32.0 / 82.9     |
| LLaVA-1.5-13B   | 0.4 / 22.9     | 7.4 / 32.3     | 14.2 / 45.5     | 1.2 / 22.0      | 5.8 / 34.9     | 17.8 / 13.8 → 54.0 / 70.8     | 31.0 / 83.5     |

> *Note: CIDEr* values can exceed 100 due to multiple scaling (see [CIDEr Paper](https://arxiv.org/abs/1411.5726)).

---

### 🧢 FACAD Fashion Captioning Results
The last two models show the worst (CNN-C) and best (SRFC) models reported by Yang et al., the best model being the one presented by the authors.

| **Model**              | **BLEU-4**  | **METEOR**   | **ROUGE-L**  | **CIDEr**    | **SPICE**    | **MAP / MAR**                | **Acc**         |
| ---------------------- | ----------- | ------------ | ------------ | ------------ | ------------ | ---------------------------- | --------------- |
| BLIP-2-2.7B            | 0.3 / 3.7   | 4.6 / 10.1   | 12.6 / 19.7  | 4.3 / 36.4   | 6.4 / 10.3   | 17.4 / 7.9 → **24.6**  / 22.5| 52.0 / **69.9** |
| BLIP-2-6.7B            | 0.3 / 3.5   | 4.6 / 9.8    | 12.9 / 19.1  | 4.1 / 34.4   | 6.4 / 9.8    | 17.7 / 7.8 → 23.3 / 21.2     | 51.4 / 69.0     |
| BLIP-2-XL              | 0.2 / 0.1   | 4.2 / 3.3    | 12.9 / 8.2   | 3.2 / 2.3    | 6.4 / 5.8    | 17.6 / 7.2 → 18.5 / 6.3      | 52.2 / 45.7     |
| BLIP-2-XXL             | 0.2 / 0.1   | 4.2 / 3.3    | 12.8 / 9.2   | 3.1 / 2.3    | 6.4 / 5.4    | 18.4 / 7.4 → 17.7 / 5.5      | 53.1 / 41.6     |
| LLaVA-1.5-7B           | 0.2 / 1.5   | 6.8 / 11.2   | 12.0 / 15.4  | 0.1 / 1.2    | 4.1 / 7.4    | 9.8 / 8.7 → 14.7 / **27.9**  | 45.1 / 65.5     |
| LLaVA-1.5-13B          | 0.2 / 0.7   | 6.5 / 8.3    | 12.4 / 12.6  | 0.8 / 0.1    | 4.5 / 3.7    | 9.9 / 8.5 → 8.4 / 16.3       | 44.2 / 27.4     |
| CNN-C (Yang et al.)    | 2.1    | 7.2       | 16.3    | 20.8    | 6.5       | 4.9                       | 10.8         |
| **SRFC (Yang et al.)** | **6.8** | **13.2** | **24.2** | **42.1** | **13.4** | 9.5                      | 18.2        |

---


## Recommendations

For the recommendations, we used the [Duche-meets-Elliot](https://github.com/sisinflab/Ducho-meets-Elliot) framework and follow the steps form the repo.

We include a script to flatten the images provided in subfolders from the H&M dataset (`flatten_images.sh`). As well as a .csv file with the generated descriptions (`generated_test_items.csv`) with which the articles were augmented.

### Results

#### 📊 Recommendation Results (Original)

| **Setting**                | NDCG\@5     | NDCG\@12    | MAP\@5      | MAP\@12     |
| -------------------------- | ----------- | ----------- | ----------- | ----------- |
| Random                     | 0.00005     | 0.00007     | 0.00005     | 0.00004     |
| MostPop                    | 0.00538     | 0.00640     | 0.00439     | 0.00373     |
| UserKNN (k=10)             | 0.01198     | 0.01355     | 0.00558     | 0.00379     |
| **ItemKNN (k=20)**         | **0.05760** | **0.06421** | **0.03557** | **0.02534** |
| Visual (ResNet50)          | 0.02414     | 0.02882     | 0.01213     | 0.00917     |
| Visual (BLIP-2)            | 0.02167     | 0.02631     | 0.01092     | 0.00840     |
| **Textual (SentenceBERT)** | **0.02509** | **0.02986** | **0.01270** | **0.00955** |
| Multimodal (ResBERT)       | 0.02427     | 0.02892     | 0.01220     | 0.00922     |
| Multimodal (CLIP)          | 0.02455     | 0.02917     | 0.01241     | 0.00934     |
| Queries (BLIP-2)           | 0.02451     | 0.02912     | 0.01247     | 0.00939     |

---

#### 📊 Recommendation Results (Unfiltered & Augmented)

| **Setting**            | NDCG\@5     | NDCG\@12    | MAP\@5      | MAP\@12     |
| ---------------------- | ----------- | ----------- | ----------- | ----------- |
| Random                 | 0.00006     | 0.00008     | 0.00005     | 0.00005     |
| MostPop                | 0.00534     | 0.00634     | 0.00437     | 0.00371     |
| UserKNN (k=10)         | 0.01214     | 0.01370     | 0.00564     | 0.00384     |
| **ItemKNN (k=20)**     | **0.05762** | **0.06423** | **0.03560** | **0.02537** |
| Visual (ResNet50)      | 0.02414     | 0.02881     | 0.01220     | 0.00921     |
| Visual (BLIP-2)        | 0.02141     | 0.02606     | 0.01080     | 0.00834     |
| Textual (SentenceBERT) | **0.02471** | **0.02950** | 0.01246     | 0.00924     |
| Multimodal (ResBERT)   | 0.02425     | 0.02887     | 0.01226     | 0.00923     |
| **Multimodal (CLIP)**  | 0.02466     | 0.02925     | **0.01254** | **0.00943** |
| Queries (BLIP-2)       | 0.02465     | 0.02914     | 0.01240     | 0.00933     |

---
