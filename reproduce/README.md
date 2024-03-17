# TASTE

### Preprocesing dataset

We provide all processed data files, which can be directly downloaded and used. If you want to repeat data processing, please refer to the following steps.

This preprocessing dataset is composed of 3 steps:
1. Get the processed dataset from Recbole/DIF-SR. We have provided this file directly, if you want to get it by yourself, please refer to data/gen_dataset_example.py. One thing worth noting: You need to modify the .inter and .item files and change their categories to token not token_seq so that they can form complete text. Please refer to [issue10](https://github.com/OpenMatch/TASTE/issues/10).
2. Run data/gen_all_items.py
3. Run data/build_train.py

We provide bash script files in reproduce/dataprocess.

Please make sure that the files under the Data folder contain the following before running:

```

data/
├── beauty/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── item.txt
│   ├── train_name.jsonl
│   ├── valid_name.jsonl
│   └── item_name.jsonl
│   
├── toys/
│  
├── sports/
│  
└── yelp/
  
```

### Checkpoint

We provide checkpoints of the four datasets that have been trained, and you can download and use them directly.

