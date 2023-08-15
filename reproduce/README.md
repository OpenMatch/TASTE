# TASTE

### Preprocesing dataset

This preprocessing dataset is composed of 3 steps:
1. Get the processed dataset from Recbole/DIF-SR. We have provided this file directly, if you want to get it by yourself, please refer to Data/gen_dataset_example.py.
2. Run Data/gen_all_items.py
3. Run Data/build_train.py

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

### Training

We provide the bash script for training in reproduce/train, as well as the tensorboard file for our own training.

### Evaluation

We provide the bash script for evaluation in reproduce/valid, as well as the tensorboard file and evaluation result log file for our own evaluation.

### Testing

We provide the bash script for the test at reproduce/test, as well as log files of our test results.


