import argparse

from src.scanner.scanner import SudokuScanner
from src.solver.plp_solver import SudokuSolver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Sudoku solving!!!')
    parser.add_argument('-i', '--input_file_path', help= 'Path to input .png file')
    parser.add_argument('-t', '--tess', help='Path to tesseract executable file')
    args = parser.parse_args()

    sudoku_scanner = SudokuScanner(args['tess'])
    sudoku_solver = SudokuSolver()

    board = sudoku_scanner.scan_board(args['input_file_path'])

    result = sudoku_solver.solve(board)

    print(result)


# Use example:
# py main.py -i ./data/sudoku_1.png -t C:\Program Files\Tesseract-OCR\tesseract.exe