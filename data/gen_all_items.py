
import sys
sys.path.append('/data1/meisen/TASTE-main')

import json
import os
from argparse import ArgumentParser
from transformers import T5Tokenizer

from utils.data_loader import load_item_name, load_item_address


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Amazon', help='choose {Amazon} or {yelp}')
    parser.add_argument('--item_file', type=str,
                        default='/data1/meisen/TASTE-main/Data/beauty/item.txt',
                        help='Path of the item.txt file')
    parser.add_argument('--output', type=str, default='item_name.jsonl')
    parser.add_argument('--output_dir', type=str, default='/data1/meisen/TASTE-main/Data/beauty',
                        help='Output data path.')
    parser.add_argument('--tokenizer', type=str,
                        default='/data1/meisen/pretrained_model/t5-base')
    parser.add_argument('--item_size', type=int, default=32,
                        help='maximum length of tokens of item text')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if args.data_name == 'Amazon':
        item_desc = load_item_name(args.item_file)
    elif args.data_name == 'yelp':
        item_desc = load_item_address(args.item_file)
    output_file = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_file, 'w') as f:
        for id, item in item_desc.items():
            group = {}
            item_ids = tokenizer.encode(item, add_special_tokens=False, padding=False, truncation=True, max_length=args.item_size)
            group['id'] = id
            group['item_ids'] = item_ids
            f.write(json.dumps(group) + '\n')

    print('-----finish------')


if __name__ == '__main__':
    main()
