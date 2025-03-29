# Check if TEMP_DIR_VGRPB is defined, otherwise use default path
if [ -z "${TEMP_DIR_VGRPB}" ]; then
  TEMP_DIR_VGRPB="../../TEMP_DIR_VGRPB"
fi

vllm serve meta-llama/Llama-3.2-90B-Vision-Instruct --port 8000 --device auto --dtype float16 --max_num_seqs 4 --enforce_eager --max-model-len 12800 --tensor-parallel-size 8 --download-dir $TEMP_DIR_VGRPB

# Replace the Qwen/Qwen2.5-VL-7B-Instruct with the following models
# meta-llama/Llama-3.2-11B-Vision-Instruct
# meta-llama/Llama-3.2-90B-Vision-Instruct
# llava-hf/llava-v1.6-mistral-7b-hf
# llava-hf/llava-onevision-qwen2-7b-ov-hf
# Qwen/Qwen2-VL-7B-Instruct
# Qwen/Qwen2-VL-72B-Instruct
# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen/Qwen2.5-VL-72B-Instruct
# Qwen/QVQ-72B-Preview