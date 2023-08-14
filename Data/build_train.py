import sys
sys.path.append('/data1/meisen/TASTE-main')
import json
import os.path
from argparse import ArgumentParser
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

from utils.data_loader import load_item_name, load_item_address, list_split


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Amazon',help='choose Amazon or yelp')
    parser.add_argument('--train_file', type=str,
                        default='/data1/meisen/TASTE-main/Data/beauty/valid.txt',
                        help='Path of the train/valid.txt file')
    parser.add_argument('--item_file', type=str,
                        default='/data1/meisen/TASTE-main/Data/beauty/item.txt',
                        help='Path of the item.txt file')
    parser.add_argument('--item_ids_file', type=str,
                        default='/data1/meisen/TASTE-main/Data/beauty/item_name.jsonl',
                        help='Path of the item token file')
    parser.add_argument('--output', type=str, default='valid_name.jsonl')
    parser.add_argument('--output_dir', type=str, default='/data1/meisen/TASTE-main/Data/beauty',
                        help='Output data path.')
    parser.add_argument('--split_num', type=int, default=243,
                        help='token num of seq text without prompt, total num equals to 256')
    parser.add_argument('--sample_num', type=int, default=100,
                        help='the sample num of random negatives ')
    parser.add_argument('--seed', type=int, default=2022,
                        help='random seed')
    parser.add_argument('--tokenizer', type=str,
                        default='/data1/meisen/pretrained_model/t5-base')
    args = parser.parse_args()
    return args





def load_item_input_ids(filename):
    item_input_ids_dict = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for example in jsonlines.Reader(f):
            id = example['id']
            item_ids = example['item_ids']
            item_input_ids_dict[id] = item_ids
    return item_input_ids_dict

def load_data(filename,item_desc):
    data = []
    data_ids = []
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        example = list()
        example2 = list()
        line = line.strip().split('\t')
        target = int(line[-1])
        seq_id = line[1:-1]
        text_list = []
        for id in seq_id:
            id = int(id)
            if id==0:
                break
            text_list.append(item_desc[id])
            example2.append(id)
        text_list.reverse()
        seq_text = ', '.join(text_list)
        example.append(seq_text)
        example.append(target)
        example2.append(target)
        data.append(example)
        data_ids.append(example2)
    return data,data_ids

def load_random_neagtive_items(args,item_num,data_num,train_data_ids):
    np.random.seed(args.seed)
    negative_samples = {}
    for i in range(data_num):
        samples = []
        for _ in range(args.sample_num):
            item = np.random.choice(item_num) + 1
            while item in train_data_ids[i] or item in samples:
                item = np.random.choice(item_num) + 1
            samples.append(item)
        negative_samples[i] = samples
    return negative_samples



def main():
    args = get_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    item_input_ids_dict = load_item_input_ids(args.item_ids_file)
    item_num = len(item_input_ids_dict)
    print('item num is %d' % item_num)
    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if args.data_name == 'Amazon':
        item_desc = load_item_name(args.item_file)
    elif args.data_name == 'yelp':
        item_desc = load_item_address(args.item_file)
    train_data, train_data_ids = load_data(args.train_file, item_desc)
    data_num = len(train_data)
    print('data num is %d' % data_num)
    random_neg_dict = load_random_neagtive_items(args, item_num, data_num, train_data_ids)
    output_file = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    template1 = "Here is the visit history list of user: "
    template2 = " recommend next item "
    t1 = tokenizer.encode(template1, add_special_tokens=False,
                          truncation=False)
    t2 = tokenizer.encode(template2, add_special_tokens=False,
                          truncation=False)
    with open(output_file, 'w') as f:
        for idx, data in enumerate(tqdm(train_data)):
            pos_list = []
            neg_list = []
            query = data[0]
            query = tokenizer.encode(query, add_special_tokens=False, padding=False, truncation=False)
            query_list = list_split(query,args.split_num)
            query_list[0] = t1 + query_list[0] + t2
            pos = data[1]
            group = {}
            pos_list.append(item_input_ids_dict[pos])
            for id in random_neg_dict[idx]:
                neg_list.append(item_input_ids_dict[id])
            group['query'] = query_list
            group['positives'] = pos_list
            group['negatives'] = neg_list
            f.write(json.dumps(group) + '\n')

    print('-----finish------')


if __name__ == '__main__':
    main()