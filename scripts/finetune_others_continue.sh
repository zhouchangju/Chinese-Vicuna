# 注意得用bash执行：bash finetune_others_continue.sh

TOT_CUDA="0"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

#DATA_PATH="sample/instruct/data_sample.jsonl" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json"
DATA_PATH="../data/0603.1/train.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json"
OUTPUT_PATH="../output/lora-Vicuna-output"
MODEL_PATH="../models/llama-7B"
lora_checkpoint="../models/Chinese-Vicuna-lora-7b-belle-and-guanaco-5800"
from_data_beginning=True # False
TEST_SIZE=2000


# 改为单显卡
CUDA_VISIBLE_DEVICES=0 python ../finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE \
--resume_from_checkpoint $lora_checkpoint \
--lora_remote_checkpoint $lora_remote_checkpoint \
--ignore_data_skip $from_data_beginning
