export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --data_name yelp  \
    --experiment_name address  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 512  \
    --best_model_path /data1/meisen/TASTE-main/checkpoint/yelp/address/best_dev    > test_yelp.out  2>&1 &
