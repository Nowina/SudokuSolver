import numpy as np
import pulp as plp


class SudokuSolver():
    # defining target value ranges in cells
    _rows = range(0,9)
    _cols = range(0,9)
    _values = range(1,10)
    _grids = range(0,9) # range of 3x3 sub_grids

    _sudoku_variables = plp.LpVariable('grid_value', (_rows, _cols, _values), cat= plp.const.LpBinary)

    def __init__(self) -> None:
        """Initialise solver with typical sudoku constraints"""
        
        self._problem = plp.LpProblem('sudoku_solver')

        # setting dummy objective
        self._problem.setObjective(plp.lpSum(0))

        self.add_constraints()
    
    def solve(self, input_values: np.ndarray) -> np.ndarray:
        problem_new = self.add_initial__values(input_values)

        return problem_new.solve()
    
    def add_initial__values(self, input_values: np.ndarray) -> plp.LpProblem:
        """Add initial values from input

        Args:
            input_values (np.ndarray): matrix containing input values
            problem (plp.LpProblem): defined plp problem

        Raises:
            TypeError: Input_values shape not equal (9,9)

        Returns:
            plp.LpProblem: problem with constraints representing input values
        """

        problem = self._problem.copy()

        if input_values.shape != (9,9):
            raise TypeError('input_values array has to be in shape (9,9)')
        
        for row in self._rows:
            for col in self._cols:
                if(input_values[row][col] != 0):
                    problem.addConstraint(plp.LpConstraint(
                        e=plp.lpSum([self._sudoku_variables[row][col]
                                    [value]*value for value in self._values]),
                        sense=plp.LpConstraintEQ,
                        rhs=input_values[row][col],
                        name=f"constraint_input_{row}_{col}"))
        
        return problem
        
    def add_constraints(self) -> None:
        """Add constraints defining sudoku

        Args:
            problem (plp.LpProblem): defined plp problem
        """
        # Add constraint to make cell hold only one value
        for row in self._rows:
            for col in self._cols:
                self._problem.addConstraint(plp.LpConstraint(
                    e=plp.lpSum([self._sudoku_variables[row][col][value]
                                 for value in self._values]),
                    sense=plp.LpConstraintEQ,
                    rhs=1,
                    name=f"constraint_one_{row}_{col}"))


        # Add constraint making _values from 1-9 to appear once in row      
        for row in self._rows:
            for value in self._values:
                self._problem.addConstraint(plp.LpConstraint(
                    e=plp.lpSum([self._sudoku_variables[row][col][value]
                                * value for col in self._cols]),
                    sense=plp.LpConstraintEQ,
                    rhs=value,
                    name=f"constraint_unique_row_{row}_{value}"))

        # Add constraint making _values from 1-9 to appear once in column       
        for col in self._cols:
            for value in self._values:
                self._problem.addConstraint(plp.LpConstraint(
                    e=plp.lpSum([self._sudoku_variables[row][col][value]
                                * value for row in self._rows]),
                    sense=plp.LpConstraintEQ,
                    rhs=value,
                    name=f"constraint_unique_col_{col}_{value}"))


        # Add constraint making _values from 1-9 to appear once in a 3x3 subgrid
        for grid in self._grids:
            grid_row  = int(grid/3)
            grid_col  = int(grid%3)

            for value in self._values:
                self._problem.addConstraint(plp.LpConstraint(
                    e=plp.lpSum([self._sudoku_variables[grid_row*3+row][grid_col*3+col][value]
                                * value for col in range(0, 3) for row in range(0, 3)]),
                    sense=plp.LpConstraintEQ,
                    rhs=value,
                    name=f"constraint_unique_grid_{grid}_{value}"))