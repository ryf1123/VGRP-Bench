from typing import List, Tuple, Union, Dict, Any
import random
import copy
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import json
import argparse
from multiprocessing import Pool, cpu_count

from puzzles.common_puzzle_factory import PuzzleFactory
from puzzles.common_constriants import Constraint, ConstraintRowNoRepeat, ConstraintColNoRepeat, ConstraintSubGridNoRepeat

class SudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        
        self.game_name = "sudoku"
        self.size = size

        self.constraints.append(ConstraintRowNoRepeat())
        self.constraints.append(ConstraintColNoRepeat())
        self.constraints.append(ConstraintSubGridNoRepeat())

        self.all_possible_values = [i for i in range(1, size + 1)]
        self.num_solver_processes = max(os.cpu_count() // 2, 1)  # Use half of available CPUs
    
    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
