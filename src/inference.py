from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import textwrap
import matplotlib.gridspec as gridspec
import re
from datasets import get_dataset_config_names
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def visualize_example(example, dataset_name, example_idx, filename):
    """
    Visualize a dataset example and save it to a file.
    
    Args:
        example: The dataset example to visualize
        dataset_name: Name of the dataset
        example_idx: Index of the example
        filename: Filename to save the visualization
        
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    try:
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Plot the prompt (text)
        ax1 = plt.subplot(gs[0, 0])
        ax1.axis('off')
        prompt_text = example.get('prompt', 'No prompt available')
        wrapped_text = textwrap.fill(prompt_text, width=80)
        ax1.text(0, 0.5, wrapped_text, fontsize=10, wrap=True)
        ax1.set_title("Prompt")
        
        # Plot the input grid (if available)
        ax2 = plt.subplot(gs[0, 1])
        if 'input_grid' in example:
            input_grid = np.array(example['input_grid'])
            ax2.imshow(input_grid, cmap='viridis')
            ax2.set_title("Input Grid")
        else:
            ax2.axis('off')
            ax2.text(0.5, 0.5, "No input grid available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot the target grid (if available)
        ax3 = plt.subplot(gs[1, 0])
        if 'target_grid' in example:
            target_grid = np.array(example['target_grid'])
            ax3.imshow(target_grid, cmap='viridis')
            ax3.set_title("Target Grid")
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, "No target grid available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot additional information
        ax4 = plt.subplot(gs[1, 1])
        ax4.axis('off')
        info_text = f"Dataset: {dataset_name}\nExample: {example_idx}\n"
        
        # Add any other fields that might be interesting
        for key, value in example.items():
            if key not in ['prompt', 'input_grid', 'target_grid'] and not isinstance(value, (list, dict, np.ndarray)):
                info_text += f"{key}: {value}\n"
        
        ax4.text(0, 0.5, info_text, fontsize=10)
        ax4.set_title("Additional Information")
        
        # Add a title to the figure
        plt.suptitle(f"Dataset: {dataset_name}, Example: {example_idx}", fontsize=16)
        
        # Save the figure
        full_path = os.path.join(output_dir, filename)
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, bbox_inches='tight')
        plt.close(fig)
        
        # Also save the raw example as JSON for reference
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Convert example to a JSON-serializable format
        example_dict = {}
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                example_dict[key] = value.tolist()
            elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                example_dict[key] = value
            else:
                example_dict[key] = str(value)
        
        with open(json_path, 'w') as f:
            json.dump(example_dict, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error visualizing example: {e}")
        return False

def get_available_subsets():
    """Get the list of available subsets for the VGRP-Bench dataset."""
    # Get all available subsets directly using the datasets library
    subsets_list = get_dataset_config_names("VGRP-Bench/VGRP-Bench")
    
    if subsets_list:
        print(f"Found {len(subsets_list)} available subsets")
        return subsets_list
    else:
        # Default fallback if no subsets are found
        print("No subsets found, using default subset")
        return ['aquarium_4x4']

def process_dataset(subset_name):
    """Process a specific dataset subset."""
    print(f"\nProcessing subset: {subset_name}")
    
    # Load the specific dataset subset
    dataset = load_dataset("VGRP-Bench/VGRP-Bench", subset_name)
    
    # Process all examples instead of just the first one
    if len(dataset) > 0:
        # The dataset is a DatasetDict with splits
        if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
            # Process each split
            for split_name, split_data in dataset.items():
                print(f"Processing split: {split_name} with {len(split_data)} examples")
                
                # Process all examples in the split
                for idx, example in enumerate(split_data):
                    # Create a filename for the image
                    filename = f"{subset_name}_{split_name}_example_{idx}.png"
                    
                    # Create and save the visualization
                    success = visualize_example(example, f"{subset_name}/{split_name}", idx, filename)
                    
                    if success:
                        print(f"Saved visualization: {filename}")
        else:
            # Dataset without splits - process all examples
            for idx, example in enumerate(dataset):
                # Create a filename for the image
                filename = f"{subset_name}_example_{idx}.png"
                
                # Create and save the visualization
                success = visualize_example(example, subset_name, idx, filename)
                
                if success:
                    print(f"Saved visualization: {filename}")
    
    print(f"Finished processing. Visualizations and data saved to '{output_dir}' directory.")

if __name__ == "__main__":
    import argparse
    import multiprocessing
    from functools import partial
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process VGRP-Bench dataset examples.')
    parser.add_argument('--subset', type=str, default='aquarium_4x4',
                        help='Dataset subset to process (default: aquarium_4x4)')
    parser.add_argument('--list', action='store_true',
                        help='List all available subsets and exit')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name to use for inference')
    parser.add_argument('--api_url', type=str, default="http://localhost:8000/v1",
                        help='API endpoint URL for the model')
    parser.add_argument('--num_workers', type=int, default=50,
                        help='Number of parallel workers for inference (default: 8)')
    
    args = parser.parse_args()
    
    # Get available subsets
    subsets_list = get_available_subsets()
    
    # If --list flag is provided, just list the subsets and exit
    if args.list:
        print("\nAvailable Dataset Subsets:")
        for subset_name in subsets_list:
            print(f"- {subset_name}")
        exit(0)
    
    # Check if the requested subset exists
    if args.subset not in subsets_list:
        print(f"Warning: Subset '{args.subset}' not found in available subsets.")
        print("Available subsets:")
        for subset_name in subsets_list:
            print(f"- {subset_name}")
        print(f"\nProceeding with '{args.subset}' anyway...")
    
    # Load the dataset
    dataset = load_dataset("VGRP-Bench/VGRP-Bench", args.subset)
    
    # Process all examples instead of just the first one
    first_split = list(dataset.keys())[0]
    examples_data = []
    
    print(f"Processing {len(dataset[first_split])} examples from split: {first_split}")
    
    # Define a function to process a single example
    def process_example(example, index, model_name=None, api_url=None):
        # Extract image, prompt, and initialization
        img = example['file_name'] if 'file_name' in example else None
        prompt = example.get('prompt', 'No prompt available')
        initialization = example.get('initialization', 'No initialization available')
        
        # Save the image to tmp directory with a random string to avoid conflicts
        image_path = None
        if img is not None:
            import uuid
            import tempfile
            
            # Create tmp directory if it doesn't exist
            tmp_dir = os.path.join('tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Generate a unique filename with random UUID
            random_id = str(uuid.uuid4())[:8]
            image_path = os.path.join(tmp_dir, f'tmp_{index}_{random_id}.png')
            
            # If it's a PIL image, save it directly
            if hasattr(img, 'save'):
                img.save(image_path)
                print(f"Image saved to {image_path}")
            # If it's a numpy array, use matplotlib
            elif isinstance(img, np.ndarray):
                plt.figure(figsize=(8, 8))
                plt.imshow(img, cmap='viridis')
                plt.axis('off')
                plt.savefig(image_path, bbox_inches='tight')
                plt.close()
                print(f"Image saved to {image_path}")
            else:
                print(f"Image type not recognized: {type(img)}")
                image_path = None
        else:
            print("No image available in the example")
        
        print(f"\nExample {index}:")
        print(f"Prompt: {prompt}")
        print(f"Initialization: {initialization}")
        
        # If a model is specified, run inference
        if model_name:
            from inference_api import generate_response
            
            print(f"\nRunning inference with model: {model_name} on example {index}")
            
            # Generate response using the inference API
            response = generate_response(
                model_name=model_name,
                prompt_text=prompt,
                image_path=image_path,
                api_url=api_url
            )
            
            print(f"\nModel Response for example {index}:")
            print("=" * 80)
            print(response)
            print("=" * 80)
            
            # Return the example and response
            return {
                "example_index": index,
                "prompt": prompt,
                "initialization": initialization,
                "response": response
            }
        
        return None
    
    if args.model:
        # Create a partial function with fixed arguments
        process_func = partial(
            process_example, 
            model_name=args.model, 
            api_url=args.api_url
        )
        
        # Prepare the arguments for multiprocessing
        examples_with_indices = [(example, idx) for idx, example in enumerate(dataset[first_split])]
        
        # Use multiprocessing to process examples in parallel
        num_workers = min(args.num_workers, len(examples_with_indices))
        print(f"Using {num_workers} workers for parallel processing")
        
        # Initialize progress bar
        pbar = tqdm(total=len(examples_with_indices), desc="Processing examples")
        
        # Define a helper function instead of using lambda
        def process_example_wrapper(args):
            example, idx = args
            return process_func(example, idx)
        
        # Create the pool and process examples
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap to process examples and get results as they complete
            results = []
            for result in pool.imap_unordered(process_example_wrapper, examples_with_indices):
                results.append(result)
                pbar.update(1)
        
        # Close the progress bar
        pbar.close()
        
        # Filter out None results and sort by example_index to maintain order
        examples_data = [r for r in results if r is not None]
        examples_data.sort(key=lambda x: x["example_index"])
        
        # Save all responses to a single JSON file
        os.makedirs(f"output/{args.subset}", exist_ok=True)
        all_responses_filename = f"output/{args.subset}/{args.model}.json"
        with open(all_responses_filename, 'w') as f:
            json.dump(examples_data, f, indent=2)
        print(f"All responses saved to {all_responses_filename}")
    else:
        # Process examples sequentially if no model is specified
        for index, example in tqdm(enumerate(dataset[first_split]), total=len(dataset[first_split]), desc="Processing examples"):
            result = process_example(example, index)
            if result:
                examples_data.append(result)
            
