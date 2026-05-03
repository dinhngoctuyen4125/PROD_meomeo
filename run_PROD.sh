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

OutputDir="outputs/results/PROD_lr${lr}"
suffix="2026"

for file in "$SaveModelPath"/*; do
filename=$(basename "$file")
echo "Filename: ${filename}, Path: ${file}"

python test_model_utility.py \
--model_name ${ModelName} \
--model_path ${file} \
--dataset "HumanEval" \
--num-samples 5 \
--acctual-num-samples 5 \
--temperature 0.2 \
--output-dir ${OutputDir}/${filename}/model_utility \
--output-file-suffix ${suffix}

python evaluatre.py \
--dataset HumanEval \
--input_path "${OutputDir}/${filename}/model_utility/HumanEval_${ModelName}_temp0.2_toppNone_topkNone_samples5_0shot_${suffix}.jsonl" \
--truncate \
--eval_standard \
--k_list 1 3 5

done