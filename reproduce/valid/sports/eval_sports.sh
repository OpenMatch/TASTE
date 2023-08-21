export CUDA_VISIBLE_DEVICES=0
nohup python evaluate.py  \
    --data_name sports  \
    --experiment_name name \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 512  \
    --stopping_step 5  \
    --all_models_path /data1/meisen/TASTE-main/checkpoint/sports/name    > eval_sports.out  2>&1 &