# 指令式微调
TOT_CUDA="2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="../data/0603.1/train.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"
OUTPUT_PATH="../output/lora-Vicuna-output-instruction"
MODEL_PATH="../models/llama-7B"
#lora_checkpoint="../models/Chinese-Vicuna-lora-7b-belle-and-guanaco-5800"
TEST_SIZE=700

# 改为单显卡
CUDA_VISIBLE_DEVICES=0 python ../finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE
