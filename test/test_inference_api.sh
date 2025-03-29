#!/bin/bash

# Test script for src/inference_api.py
# This script tests the gpt-4o model with a single prompt and images

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test cases - using all models from the configuration file
# for full list, please refer to configs/full_list_model.txt
declare -a models=(
    # "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "gpt-4o"
    # "claude-3-5-sonnet-20241022"
)
declare -a prompts=(
    "Describe what you see in this image.",
)

# Run tests
echo -e "${GREEN}Starting tests...${NC}"

# Function to run a test
run_test() {
    local model=$1
    local prompt=$2
    local image=$3
    local api_url="http://localhost:8000/v1"
    
    echo -e "${BLUE}Testing model: ${model}${NC}"
    echo -e "${BLUE}Prompt: ${prompt}${NC}"
    echo -e "${BLUE}Image: ${image}${NC}"

    # Run the inference
    if [ -z "$image" ]; then
        python src/inference_api.py --model "$model" --prompt "$prompt" --api_url "$api_url"
    else
        python src/inference_api.py --model "$model" --prompt "$prompt" --image "$image" --api_url "$api_url"
    fi
    
    echo -e "${GREEN}Test completed${NC}"
    echo "----------------------------------------"
}

# Test without images
for model in "${models[@]}"; do
    run_test "$model" "Explain the concept of recursion in programming." ""
done

# Test with images - only using the first prompt
for model in "${models[@]}"; do
    run_test "$model" "${prompts[0]}" "assets/Cat03.jpg"
    # run_test "$model" "${prompts[0]}" "assets/CodeCmmt002.svg.png"
    # run_test "$model" "${prompts[0]}" "assets/chart.png"
done

echo -e "${GREEN}All tests completed!${NC}" 