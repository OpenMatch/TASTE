python build_train.py  \
    --data_name yelp  \
    --train_file /data1/meisen/TASTE-main/Data/yelp/train.txt  \
    --item_file /data1/meisen/TASTE-main/Data/yelp/item.txt  \
    --item_ids_file /data1/meisen/TASTE-main/Data/yelp/item_address.jsonl  \
    --output train_address.jsonl  \
    --output_dir /data1/meisen/TASTE-main/Data/yelp  \
    --seed 2022  \
    --tokenizer t5-base   

python build_train.py  \
    --data_name yelp  \
    --train_file /data1/meisen/TASTE-main/Data/yelp/valid.txt  \
    --item_file /data1/meisen/TASTE-main/Data/yelp/item.txt  \
    --item_ids_file /data1/meisen/TASTE-main/Data/yelp/item_address.jsonl  \
    --output valid_address.jsonl  \
    --output_dir /data1/meisen/TASTE-main/Data/yelp  \
    --seed 2022  \
    --tokenizer t5-base   

python build_train.py  \
    --data_name Amazon  \
    --train_file /data1/meisen/TASTE-main/Data/beauty/train.txt  \
    --item_file /data1/meisen/TASTE-main/Data/beauty/item.txt  \
    --item_ids_file /data1/meisen/TASTE-main/Data/beauty/item_name.jsonl  \
    --output train_name.jsonl  \
    --output_dir /data1/meisen/TASTE-main/Data/beauty  \
    --seed 2022  \
    --tokenizer t5-base   

python build_train.py  \
    --data_name Amazon  \
    --train_file /data1/meisen/TASTE-main/Data/beauty/valid.txt  \
    --item_file /data1/meisen/TASTE-main/Data/beauty/item.txt  \
    --item_ids_file /data1/meisen/TASTE-main/Data/beauty/item_name.jsonl  \
    --output valid_name.jsonl  \
    --output_dir /data1/meisen/TASTE-main/Data/beauty  \
    --seed 2022  \
    --tokenizer t5-base  
 