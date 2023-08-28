export CUDA_VISIBLE_DEVICES=3
nohup python evaluate.py  \
    --data_name beauty  \
    --experiment_name name \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 512  \
    --stopping_step 5  \
    --all_models_path /data1/TASTE/checkpoint/beauty/name    > eval_beauty.out  2>&1 &