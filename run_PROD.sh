#!/bin/bash

lr=5e-6

Model="deepseek-ai/deepseek-coder-1.3b-instruct"
ModelName="Deepseek-1.3b"
ModelPath="deepseek-ai/deepseek-coder-1.3b-instruct"
DatasetPath="data/depAPI.json"
SaveModelPath="outputs/models/PROD_lr${lr}"

python PROD.py \
--model_name ${Model} \
--model_path ${ModelPath} \
--output_dir ${SaveModelPath} \
--train_data_path ${DatasetPath} \
--alpha 0.0 \
--num_train_epochs 10 \
--learning_rate ${lr} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--logging_steps 1 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_train \
--save_strategy no || exit