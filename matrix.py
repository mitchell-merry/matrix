from __future__ import annotations

class Matrix:
    def __init__(self, default: list[list[int]]):

        # Matrix must be a double array where rows and cols are > 0
        if not isinstance(default, list) or len(default) == 0 or not isinstance(default[0], list) or len(default[0]) == 0:
            raise ValueError("Invalid matrix!")

        self.matrix = default
    
    def append(self, other: Matrix) -> Matrix:
        """Appends one matrix onto the right of another.

        Rows must match in size. Does not modify the original matrix in-place.

        Returns a new matrix.
        """

        if len(self.matrix) != len(other.matrix):
            raise ValueError("Matrices must have the same number of rows!")

        return Matrix([
            self.copyRowByIndex(row) + other.copyRowByIndex(row) for row, _ in enumerate(self.matrix)
        ])

    def interchange(self, rowI: int, rowJ: int) -> Matrix:
        """Swaps two rows. Returns a new matrix."""

        newM = self.copy()
        newM.matrix[rowI] = self.copyRowByIndex(rowJ)
        newM.matrix[rowJ] = self.copyRowByIndex(rowI)
        
        return newM

    def swap(self, rowI: int, rowJ: int) -> Matrix: 
        """Swaps two rows. Returns a new matrix. Alias of Matrix#interchange()."""
        return self.interchange(rowI, rowJ)

    def copyRow(self, row: list[int]) -> list[int]:
        return [ x for x in row ]

    def copyRowByIndex(self, i: int) -> list[int]:
        return [ x for x in self.matrix[i] ]

    def copy(self):
        """Returns a deep copy of the matrix."""
        
        copy = Matrix([
            [ x for x in row ] for row in self.matrix
        ])

        return copy

    def print(self):
        """Prints the matrix."""

        for row in self.matrix:
            print("[", end="")
            for cell in row:
                print(f" {cell} ", end="")
            print("]")

if __name__ == "__main__":
    m = Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    m2 = Matrix([
        [1], [2], [3]
    ])


    m.append(m2).append(m2).swap(1, 2).interchange(0, 2).print()