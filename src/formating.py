import json
import argparse
import os
from tqdm import tqdm
import sys
import multiprocessing
from functools import partial
sys.path.append(".")  # Add current directory to path
from src.inference_api import generate_response, AIModelConfig, Prompt
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="gpt-4o-mini", 
                        help='Model name for input/output files')
    parser.add_argument('--subset', type=str, default="aquarium_aquarium_4x4", 
                        help='Subset name (e.g., aquarium_aquarium_4x4)')
    parser.add_argument('--model_id', type=str, default="gpt-4o-2024-11-20", 
                        help='Model ID to use for formatting')
    parser.add_argument('--num_processes', type=int, default=50,
                        help='Number of parallel processes to use')

    return parser.parse_args()


def process_one_text(model_id, text, prompt, api_url=None):
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a new conversation with the existing prompt messages
    conversation = prompt.copy()
    
    # Add the user message with the new text
    conversation.append({"role": "user", "content": f"That's right. I am giving you a new one. {text}"})
    
    # Generate response using the OpenAI Chat API
    response = client.chat.completions.create(
        model=model_id,  # Use the provided model_id (e.g., "gpt-4o")
        messages=conversation
    )
    
    # Extract the assistant's response
    formatted_response = response.choices[0].message.content
    
    print(formatted_response)
    return formatted_response

def process_one_entry(entry, model_id, prompt, api_url=None):
    model_output = entry['response']
    filtered_output = process_one_text(model_id, model_output, prompt, api_url)
    entry['model_output_filtered'] = filtered_output
    return entry

def process_json_file(input_file, output_file, prompt_file, model_id, num_processes=12):
    # Check if the model is locally deployed and set base_url
    api_url = ""
    
    # Read input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Load prompt file
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)
    
    # Create a partial function with fixed arguments
    process_func = partial(process_one_entry, model_id=model_id, prompt=prompt, api_url=api_url)
    
    # Process entries in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, data), total=len(data)))
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    # Construct file paths
    input_file = f"output/{args.subset}/{args.model}.json"
    output_file = f"output-formatted/{args.subset}/{args.model}.json"
    
    # Get the game type from subset (e.g., "aquarium" from "aquarium_4x4")
    game_type = args.subset.split("_")[0]
    prompt_file = f"configs/formating-prompt/{game_type}/filter_prompt.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Processing {input_file} -> {output_file}")
    print(f"Using prompt file: {prompt_file}")
    print(f"Using model: {args.model_id} for formatting")
    print(f"Using {args.num_processes} parallel processes")
    
    process_json_file(input_file, output_file, prompt_file, args.model_id, args.num_processes)
    