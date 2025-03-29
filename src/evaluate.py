import json
import argparse
import os
import re
from tqdm import tqdm
import numpy as np
from json_repair import repair_json

def load_all_puzzles(input_file):
    """
    Load all puzzles and create a lookup dictionary using index as id.
    
    Args:
        input_file (str): Path to the JSON file containing puzzles
        
    Returns:
        dict: Dictionary mapping puzzle IDs (as strings) to puzzle data
    """
    with open(input_file, 'r') as f:
        puzzles = json.load(f)
        return {str(idx): puzzle for idx, puzzle in enumerate(puzzles)}

def extract_perception_and_answer(model_output, args):
    """
    Extract both perception and answer from model output.
    
    Parses the model's output to extract the perceived initial state and the solution.
    Handles different output formats and section headers.
    
    Args:
        model_output (str): The raw output from the model
        args (argparse.Namespace): Command line arguments
        
    Returns:
        tuple: (initial_state, solution) where both are 2D arrays or None if parsing fails
    """

    try:
        # Handle plain text format
        if "Initial State" in model_output:
            parts = model_output.split('Initial State\n', 1)
        elif "Perception" in model_output:
            parts = model_output.split('Perception\n', 1)
        else:
            return None, None

        if len(parts) != 2:
            return None, None
        
        content = parts[1]

        if "Answer" in content:
            perception_answer = content.split('\nAnswer\n')
        elif "Solution" in content:
            perception_answer = content.split('\nSolution\n')
        else:
            return None, None
        
        if len(perception_answer) != 2:
            return None, None
            
        perception, answer = perception_answer
        
        if perception.strip() == "Wrong":
            if not args.text_input:
                return None, None
            else:
                initial_state = None
                # Remove outer brackets and split into rows
                raw_solution = answer.strip()[2:-2].split('],[')
                solution = [[c for c in row.split(',')] for row in raw_solution]
        else:
            # Remove outer brackets and split into rows
            raw_perception = perception.strip()[2:-2].split('],[')
            initial_state = [[c for c in row.split(',')] for row in raw_perception]
            raw_solution = answer.strip()[2:-2].split('],[')
            solution = [[c for c in row.split(',')] for row in raw_solution]

        if args.text_input:
            initial_state = None
        else:
            initial_state = [[cell if cell != '*' else 0 for cell in row] for row in initial_state]

        return initial_state, solution
        
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None, None

def check_perception(thoughts, init_board, game_type):
    """
    Check if model's perception matches the initial board.
    
    Compares the model's understanding of the initial state with the actual initial state,
    with game-specific adjustments for different puzzle types.
    
    Args:
        thoughts (list): 2D array representing the model's perception of the initial state
        init_board (list): 2D array representing the actual initial state
        game_type (str): Type of puzzle game
        
    Returns:
        bool: True if perception matches initial board, False otherwise
    """

    if game_type == "battleships":
        init_board = [[0 if cell == 'e' else cell for cell in row] for row in init_board]
        thoughts = [[0 if cell == 'e' else cell for cell in row] for row in thoughts]
    if game_type == "lightup":
        for i in range(len(init_board)):
            for j in range(len(init_board[i])):
                cell = init_board[i][j]
                # Check if cell is a number (not 0) or not a string/character
                if (isinstance(cell, (int, float)) and cell != 0) or (isinstance(cell, str) and not cell.isalpha()):
                    init_board[i][j] = 'w'
    if game_type == "fieldexplore":
        # init_board's -1 to 0
        init_board = [[0 if cell == -1 else cell for cell in row] for row in init_board]
        
    if isinstance(init_board, str):
        init_grid = [[c for c in row] for row in init_board.strip().split('\n')]
    else:
        init_grid = init_board

    if len(thoughts) != len(init_grid) or any(len(row) != len(init_grid[0]) for row in thoughts):
        return False
        
    for i in range(len(init_grid)):
        for j in range(len(init_grid[0])):
            if str(init_grid[i][j]) != str(thoughts[i][j]):
                return False
    return True

def check_answer(answer, init_board, game_factory):
    """
    Verify if the model's answer is correct for the given puzzle.
    
    Performs game-specific validations and uses the game factory to check solution correctness.
    
    Args:
        answer (list): 2D array representing the model's solution
        init_board (list): 2D array representing the initial state
        game_factory (GameFactory): Factory object for the specific game type
        
    Returns:
        bool: True if the answer is correct, False otherwise
    """

    global GRID_SIZE
    
    # replace 0 to 'e' in trees and tents; use a for loop to replace
    if game_factory.game_name in ["treesandtents", "starbattle", "hitori", "aquarium", "kakurasu"]:
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                if answer[i][j] in [0, '0']:
                    answer[i][j] = 'e'
    if game_factory.game_name == "oddevensudoku":
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                try:
                    answer[i][j] = int(answer[i][j])
                except Exception as e:
                    return False
    if game_factory.game_name == "lightup":
        # replace '0' to 'e'
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                if answer[i][j] == '0':
                    answer[i][j] = 'e'
    # init_board
    if isinstance(init_board, str):
        init_grid = [[c for c in row] for row in init_board.strip().split('\n')]
    else:
        init_grid = init_board
    
    # Check dimensions
    if len(answer) != GRID_SIZE or any(len(row) != GRID_SIZE for row in answer):
        return False

    # Check initial numbers preserved
    if game_factory.game_name == "hitori":
        # compare with game_factory.additional_board
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if game_factory.additional_board[i][j] not in [0, '0'] and str(game_factory.additional_board[i][j]) != str(answer[i][j]):
                    return False
    elif game_factory.game_name == "nonogram":
        # convert 0 in answer to 'e'
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if answer[i][j] in [0, '0', '*']:
                    answer[i][j] = 'e'
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if init_grid[i][j] not in [0, '0'] and str(init_grid[i][j]) != str(answer[i][j]):
                    return False
    elif game_factory.game_name == "fieldexplore":
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # s on the initial board must be kept
                if init_grid[i][j] == 's' and not answer[i][j] == 's':
                    return False
                try:
                    cell_value = int(init_grid[i][j])
                    if cell_value > 0 and str(answer[i][j]) == 's':
                        return False
                except (ValueError, TypeError):
                    # Cell is not a number, continue with other checks
                    pass
        return True
    else:
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if init_grid[i][j] not in [0, '0', 'e'] and str(init_grid[i][j]) != str(answer[i][j]):
                    return False
    
    game_state = {
        "board": answer,
        "size": GRID_SIZE,
    }

    if game_factory.game_name == "skyscraper":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "coloredsudoku":
        game_state["colors"] = game_factory.current_colors
    elif game_factory.game_name == "futoshiki":
        game_state["inequalities"] = game_factory.current_inequalities
    elif game_factory.game_name == "killersudoku":
        game_state["cages"] = game_factory.cages
    elif game_factory.game_name == "renzoku":
        game_state["hints"] = game_factory.hints
    elif game_factory.game_name == 'kakuro':
        game_state["sums"] = game_factory.current_sums
    elif game_factory.game_name == "thermometers":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "treesandtents":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "starbattle":
        game_state["regions"] = game_factory.regions
    elif game_factory.game_name == "hitori":
        game_state["numbers"] = game_factory.numbers
    elif game_factory.game_name == "aquarium":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "kakurasu":
        game_state["clues"] = game_factory.clues
    elif game_factory.game_name == "oddevensudoku":
        game_state["cell_types"] = game_factory.cell_types
    elif game_factory.game_name == "nonogram":
        game_state["hints"] = game_factory.hints
    elif game_factory.game_name == "lightup":
        game_state["wall_numbers"] = game_factory.wall_numbers
    elif game_factory.game_name == "battleships":
        game_state["hints"] = game_factory.hints
    
    try:
        return game_factory.check(game_state)
    except Exception as e:
        print(f"Error checking answer: {e}")
        return False

def calculate_group_statistics(outcomes, num_groups=5):
    """
    Calculate group-wise means and the standard deviation between groups.
    
    Splits outcomes into groups and calculates statistics to estimate variance.
    
    Args:
        outcomes (list): Binary outcomes (0 or 1) for each puzzle
        num_groups (int): Number of groups to split the data into
        
    Returns:
        tuple: (group_means, group_std) where group_means is a list of percentages 
               and group_std is the standard deviation between groups
    """
    if not outcomes:
        return [], 0.0
    
    # Convert to numpy array for easier manipulation
    outcomes = np.array(outcomes)
    
    # Calculate number of items per group
    group_size = len(outcomes) // num_groups
    
    # Split into groups and calculate mean for each group
    group_means = []
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < num_groups - 1 else len(outcomes)
        group = outcomes[start_idx:end_idx]
        group_means.append(np.mean(group) * 100)  # Convert to percentage
    
    # Calculate standard deviation between group means
    group_std = np.std(group_means)
    
    return group_means, group_std

GRID_SIZE = None  # Global variable to store the puzzle grid size

def main():
    """
    Main function to evaluate puzzle solutions.
    
    Processes model outputs for a specific puzzle subset, evaluates perception and solution accuracy,
    and calculates statistics on model performance.
    """
    parser = argparse.ArgumentParser(description='Evaluate puzzle solutions')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", 
                        help='Model name (e.g., gpt-4o-mini)')
    parser.add_argument('--subset', type=str, default="aquarium_aquarium_4x4", 
                        help='Subset name (e.g., aquarium_aquarium_4x4)')
    parser.add_argument("--thinking_format", type=str, default="json", 
                        help="Thinking format")
    parser.add_argument("--text_input", action="store_true", 
                        help="Use text input")
    
    args = parser.parse_args()
    
    # Extract game_type from subset (e.g., "aquarium" from "aquarium_aquarium_4x4")
    game_type = args.subset.split('_')[0]
    
    # Construct output file path where model results are stored
    output_file = f"output-formatted/{args.subset}/{args.model_name}.json"

    # Import game-specific modules
    import puzzles.common_get_prompt as get_prompt
    import puzzles.common_get_game_factory as get_game_factory
    
    # Initialize the appropriate game factory for the puzzle type
    GameFactory = get_game_factory.get_game_factory(game_type)
    game_factory = GameFactory(size=4)

    # Load model outputs from previous runs
    with open(output_file, 'r') as f:
        results = json.load(f)

    # Initialize counters and tracking variables
    total = len(results)
    correct_perceptions = 0
    correct_answers = 0
    answers_with_correct_perception = 0

    # Create lists to store binary outcomes for each puzzle
    perception_outcomes = []  # 1 if perception correct, 0 otherwise
    answer_outcomes = []      # 1 if answer correct, 0 otherwise
    answer_with_perception_outcomes = []  # 1 if answer correct given correct perception, 0 otherwise

    # Process each puzzle result
    for result in tqdm(results):
        puzzle_id = str(result['example_index'])
        model_output = result['model_output_filtered']
        
        # Parse initialization data from the result
        puzzle_data = eval(result['initialization'])  # Convert string representation to Python object
        init_board = puzzle_data['initialization']
        
        # Calculate number of hints (non-empty cells) in the initial board
        hint_count = sum(1 for row in init_board for cell in row if cell not in [0, '0', '*'])
        result['hint_count'] = hint_count

        # Handle game-specific initialization data
        if game_type == "coloredsudoku":
            colors = puzzle_data.get('colors', None)
            if colors is None:
                print(f"Warning: No colors found for puzzle {puzzle_id}")
                continue
            # Set colors in game factory for this puzzle
            game_factory.current_colors = colors
        elif game_type == "binairo":
            # Get clues from puzzle data
            init_board = puzzle_data.get('initialization', None)
        elif game_type == "futoshiki":
            # Get inequalities from puzzle data
            row_inequalities = puzzle_data.get('row_inequalities', None)
            col_inequalities = puzzle_data.get('col_inequalities', None)
            if row_inequalities is None or col_inequalities is None:
                print(f"Warning: No inequalities found for puzzle {puzzle_id}")
                continue
            # Set inequalities in game state
            game_factory.current_inequalities = {
                "row": row_inequalities,
                "col": col_inequalities
            }
        elif game_type == "killersudoku":
            # Get clues from puzzle data
            cages = puzzle_data.get('cages', None)
            if cages is None:
                print(f"Warning: No clues found for puzzle {puzzle_id}")
                continue
            # Set clues in game state
            game_factory.cages = cages
        elif game_type == "renzoku":
            hints = puzzle_data.get('hints', None)
            if hints is None:
                print(f"Warning: No hints found for puzzle {puzzle_id}")
                continue
            # Set hints in game state
            game_factory.hints = hints
        elif game_type == "kakuro":
            # Get sums from puzzle data
            sums = puzzle_data.get('sums', None)
            if sums is None:
                print(f"Warning: No sums found for puzzle {puzzle_id}")
                continue
            # Set sums in game state
            game_factory.current_sums = sums
        elif game_type == "skyscraper":
            # Get clues from puzzle data
            clues = puzzle_data.get('initialization', None).get('clues')
            init_board = puzzle_data.get('initialization', None).get('board') # a little special
            if clues is None:
                print(f"Warning: No clues found for puzzle {puzzle_id}")
                continue
            # Set clues in game state
            game_factory.clues = clues
        elif game_type == "thermometers":
            # Set clues in game state
            clues = puzzle_data.get('initialization', None).get('clues')
            game_factory.clues = clues
            init_board = puzzle_data.get('initialization', None).get('board')
        elif game_type == "treesandtents":
            # Set clues in game state
            clues = puzzle_data.get('clues', None)
            game_factory.clues = clues
            init_board = puzzle_data.get('initialization', None)
        elif game_type == "starbattle":
            # Set board in game state
            init_board = puzzle_data.get('initialization', None)
            game_factory.regions = puzzle_data.get('regions', None)
        elif game_type == "hitori":
            init_board = puzzle_data.get('initialization').get('numbers', None)
            game_factory.numbers = puzzle_data.get('initialization', None).get('numbers')
            game_factory.additional_board = puzzle_data.get('initialization', None).get('board')
        elif game_type == "aquarium":
            init_board = puzzle_data.get('initialization', None).get('board')
            game_factory.clues = puzzle_data.get('initialization', None).get('clues', None)
        elif game_type == "kakurasu":
            init_board = puzzle_data.get('initialization', None).get('board')
            game_factory.clues = puzzle_data.get('initialization', None).get('clues', None)
        elif game_type == "oddevensudoku":
            game_factory.cell_types = puzzle_data.get('cell_types')
            init_board = puzzle_data.get('initialization', None)
        elif game_type == "battleships":
            init_board = puzzle_data.get('initialization', None)
            game_factory.hints = puzzle_data.get('hints', None)
        elif game_type == "jigsawsudoku":
            init_board = puzzle_data.get('initialization', None)
        elif game_type == "nonogram":
            init_board = puzzle_data.get('initialization', None)
            game_factory.hints = puzzle_data.get('hints', None)
        elif game_type == "lightup":
            init_board = puzzle_data.get('initialization', None)
            game_factory.wall_numbers = puzzle_data.get('wall_numbers', None)

        # Set global grid size based on the first puzzle processed
        global GRID_SIZE
        GRID_SIZE = len(init_board) if GRID_SIZE is None else GRID_SIZE

        # Extract model's perception and answer from its output
        thoughts, answer = extract_perception_and_answer(model_output, args)
        
        if not args.text_input:
            try:
                for i in range(len(thoughts)):
                    for j in range(len(thoughts[i])):
                        if thoughts[i][j] == "*":
                            thoughts[i][j] = "0"
            except Exception as e:
                print(f"starbattle: Error converting thoughts to 0: {e}")
        
        try:
            if game_type == "killersudoku":
                answer = [[int(cell) for cell in row] for row in answer]
        except Exception as e:
            print(f"killersudoku: Error converting answer to int: {e}")
            answer = None
        
        if (thoughts is None or answer is None) and not args.text_input:
            continue
        
        if args.text_input:
            if answer is None:
                continue
        
        # convert 't' to 'tt' and 'r' to tr when trees and tents are used
        if game_type == "treesandtents":
            if not args.text_input:
                # Convert each cell in the 2D array using nested for loops
                for i in range(len(thoughts)):
                    for j in range(len(thoughts[i])):
                        if thoughts[i][j] == 't':
                            thoughts[i][j] = 'tt'
                        elif thoughts[i][j] == 'r':
                            thoughts[i][j] = 'tr'
            
            for i in range(len(answer)):
                for j in range(len(answer[i])):
                    if answer[i][j] == 't':
                        answer[i][j] = 'tt'
                    elif answer[i][j] == 'r':
                        answer[i][j] = 'tr'

        perception_correct = check_perception(thoughts, init_board, game_type) if not args.text_input else True

        if perception_correct or args.text_input:
            correct_perceptions += 1
            
            if check_answer(answer, init_board, game_factory):
                correct_answers += 1
                answers_with_correct_perception += 1
        
        perception_correct = check_perception(thoughts, init_board, game_type) if not args.text_input else True
        
        answer_correct = check_answer(answer, init_board, game_factory) if (perception_correct or args.text_input) else False

        perception_outcomes.append(1 if perception_correct else 0)
        answer_outcomes.append(1 if answer_correct else 0)
        if perception_correct:
            answer_with_perception_outcomes.append(1 if answer_correct else 0)

    # Calculate overall statistics
    correct_perceptions = sum(perception_outcomes)
    correct_answers = sum(answer_outcomes)
    answers_with_correct_perception = sum(answer_with_perception_outcomes)

    # Calculate accuracy percentages
    perception_accuracy = (correct_perceptions / total) * 100
    overall_answer_accuracy = (correct_answers / total) * 100
    answer_accuracy_with_correct_perception = (answers_with_correct_perception / len(answer_with_perception_outcomes) * 100) if answer_with_perception_outcomes else 0

    # Calculate group statistics for error estimation
    perception_group_means, perception_group_std = calculate_group_statistics(perception_outcomes)
    answer_group_means, answer_group_std = calculate_group_statistics(answer_outcomes)
    answer_with_perception_group_means, answer_with_perception_group_std = calculate_group_statistics(answer_with_perception_outcomes)

    # Print results to console
    print(f"\nResults:")
    print(f"Total puzzles: {total}")
    print(f"Perception Accuracy: {perception_accuracy:.2f}% ± {perception_group_std:.2f}%")
    print(f"Overall Answer Accuracy: {overall_answer_accuracy:.2f}% ± {answer_group_std:.2f}%")
    print(f"Answer Accuracy with Correct Perception: {answer_accuracy_with_correct_perception:.2f}% ± {answer_with_perception_group_std:.2f}%")

    # Create directories for results if they don't exist
    os.makedirs(f"results-txt", exist_ok=True)
    os.makedirs(f"results-txt/{game_type}", exist_ok=True)

    # Write results to a text file
    with open(f"results-txt/{game_type}/evaluate_results.txt", "a") as f:
        f.write(f"output_file: {output_file}\n")
        f.write(f"Total puzzles: {total}\n")
        f.write(f"Perception Accuracy: {perception_accuracy:.2f}% ± {perception_group_std:.2f}%\n")
        f.write(f"Overall Answer Accuracy: {overall_answer_accuracy:.2f}% ± {answer_group_std:.2f}%\n")
        f.write(f"Answer Accuracy with Correct Perception: {answer_accuracy_with_correct_perception:.2f}% ± {answer_with_perception_group_std:.2f}%\n")

    # Create path for augmented results JSON
    output_path = output_file.replace('.json', '_evaluation.json')

    # Add evaluation results to each puzzle result
    for i, result in enumerate(results):
        # Get initialization directly from the result
        puzzle_data = eval(result['initialization'])
        
        # Handle game-specific initialization extraction
        if puzzle_data:
            if game_type == "coloredsudoku":
                init_board = puzzle_data['initialization']
            elif game_type == "binairo":
                init_board = puzzle_data['initialization']
            elif game_type == "futoshiki":
                init_board = puzzle_data['initialization']
            elif game_type == "killersudoku":
                init_board = puzzle_data['initialization']
            elif game_type == "renzoku":
                init_board = puzzle_data['initialization']
            elif game_type == "kakuro":
                init_board = puzzle_data['initialization']
            elif game_type == "skyscraper":
                init_board = puzzle_data['initialization']['board']
            elif game_type == "thermometers":
                init_board = puzzle_data['initialization']['board']
            elif game_type == "treesandtents":
                init_board = puzzle_data['initialization']
            elif game_type == "starbattle":
                init_board = puzzle_data['initialization']
            elif game_type == "hitori":
                init_board = puzzle_data['initialization']['numbers']
            elif game_type == "aquarium":
                init_board = puzzle_data['initialization']['board']
            elif game_type == "kakurasu":
                init_board = puzzle_data['initialization']['board']
            elif game_type == "oddevensudoku":
                init_board = puzzle_data['initialization']
            elif game_type == "battleships":
                init_board = puzzle_data['initialization']
            elif game_type == "jigsawsudoku":
                init_board = puzzle_data['initialization']
            elif game_type == "nonogram":
                init_board = puzzle_data['initialization']
            elif game_type == "lightup":
                init_board = puzzle_data['initialization']
            else:
                init_board = puzzle_data['initialization']
        else:
            init_board = []
            
        # Calculate number of hints
        hint_count = sum(1 for row in init_board for cell in row if cell not in [0, '0', '*'])
        
        # Add evaluation results to the result object
        result['perception_correct'] = bool(perception_outcomes[i]) if i < len(perception_outcomes) else False
        result['answer_correct'] = bool(answer_outcomes[i]) if i < len(answer_outcomes) else False
        result['hint_count'] = hint_count

    # Save augmented results to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
