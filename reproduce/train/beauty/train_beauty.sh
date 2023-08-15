 export CUDA_VISIBLE_DEVICES=2
 nohup python train.py  \
    --output_dir /data1/meisen/TASTE-main/checkpoint/beauty/name  \
    --model_name_or_path /data1/meisen/pretrained_model/t5-base  \
    --do_train  \
    --save_steps 5000  \
    --eval_steps 5000  \
    --train_path /data1/meisen/TASTE-main/Data/beauty/train_name.jsonl  \
    --eval_path /data1/meisen/TASTE-main/Data/beauty/valid_name.jsonl  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 8  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir /data1/meisen/TASTE-main/checkpoint/beauty/name-log    > train_beauty.out  2>&1 &   