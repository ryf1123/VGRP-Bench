def get_game_factory(game_type):
    if game_type == "sudoku":
        from puzzles.sudoku import SudokuPuzzleFactory as GameFactory
    elif game_type == "binairo":
        from puzzles.binairo import BinairoPuzzleFactory as GameFactory
    elif game_type == "coloredsudoku":
        from puzzles.coloredsudoku import ColoredSudokuPuzzleFactory as GameFactory
    elif game_type == "kakuro":
        from puzzles.kakuro import KakuroPuzzleFactory as GameFactory
    elif game_type == "killersudoku":
        from puzzles.killersudoku import KillerSudokuPuzzleFactory as GameFactory
    elif game_type == "renzoku":
        from puzzles.renzoku import RenzokuPuzzleFactory as GameFactory
    elif game_type == "skyscraper":
        from puzzles.skyscraper import SkyscraperPuzzleFactory as GameFactory
    elif game_type == "starbattle":
        from puzzles.starbattle import StarBattlePuzzleFactory as GameFactory
    elif game_type == "treesandtents":
        from puzzles.treesandtents import TreesAndTentsPuzzleFactory as GameFactory
    elif game_type == "thermometers":
        from puzzles.thermometers import ThermometersPuzzleFactory as GameFactory
    elif game_type == "futoshiki":
        from puzzles.futoshiki import FutoshikiPuzzleFactory as GameFactory
    elif game_type == "hitori":
        from puzzles.hitori import HitoriPuzzleFactory as GameFactory
    elif game_type == "aquarium":
        from puzzles.aquarium import AquariumPuzzleFactory as GameFactory
    elif game_type == "kakurasu":
        from puzzles.kakurasu import KakurasuPuzzleFactory as GameFactory
    elif game_type == "oddevensudoku":
        from puzzles.oddevensudoku import OddEvenSudokuPuzzleFactory as GameFactory
    elif game_type == "battleships":
        from puzzles.battleships import BattleshipsPuzzleFactory as GameFactory
    elif game_type == "fieldexplore":
        from puzzles.fieldexplore import FieldExplorePuzzleFactory as GameFactory
    elif game_type == "jigsawsudoku":
        from puzzles.jigsawsudoku import JigsawSudokuPuzzleFactory as GameFactory
    elif game_type == "lightup":
        from puzzles.lightup import LightUpPuzzleFactory as GameFactory
    elif game_type == "nonogram":
        from puzzles.nonogram import NonogramPuzzleFactory as GameFactory
        
    return GameFactory