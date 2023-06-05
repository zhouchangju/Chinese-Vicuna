TOT_CUDA="0,1,2,3" #Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs, for example: pip install bitsandbytes==0.39.0
BASE_MODEL="../models/llama-7B" #"decapoda-research/llama-13b-hf"
LORA_PATH="../output/lora-Vicuna-output-instruction" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=0 # 1: use local model, 0: use huggingface model
TYPE_WRITER=1 # whether output streamly
if [[ USE_LOCAL -eq 1 ]]
then
cp sample/instruct/adapter_config.json $LORA_PATH
fi

#Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs
# ¸ÄÎªµ¥ÏÔ¿¨
CUDA_VISIBLE_DEVICES=0  python ../generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL \
    --use_typewriter $TYPE_WRITER