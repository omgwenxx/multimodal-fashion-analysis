
### Create files
The processed feature folders will look like the following

```
├── hm_folder
│   ├── visual_embeddings_indexed_256
|       ├── torch
|          ├── ResNet50
|             ├── avgpool
│                ├── 0.npy
│                ├── 1.npy
│                ├── ...
|       ├── transformers
|          ├── openai
|             ├── clip-vit-base-patch16
|                ├── 1
│                   ├── 0.npy
│                   ├── 1.npy
│                   ├── ...
|
│   ├── textual_embeddings_indexed_32
|       ├── sentence_transformers
|          ├── sentence-transformers
|             ├── all-mpnet-base-v2
|                ├── 1
│                   ├── 0.npy
│                   ├── 1.npy
│                   ├── ...
|       ├── transformers
|          ├── openai
|             ├── clip-vit-base-patch16
|                ├── 1
│                   ├── 0.npy
│                   ├── 1.npy
│                   ├── ...
│   ├── train_indexed.tsv
│   ├── test_indexed.tsv
```