class Matrix:
    def __init__(self, default):
        # Matrix must be a double array where rows and cols are > 0
        if not isinstance(default, list) or len(default) == 0 or not isinstance(default[0], list) or len(default[0]) == 0:
            raise ValueError("Invalid matrix!")

        self.matrix = default
    
    def append(self, other):
        """Appends one matrix onto the right of another.

        Rows must match in size. Does not modify the original matrix in-place.

        Returns a new matrix.
        """

        if len(self.matrix) != len(other.matrix):
            raise ValueError("Matrices must have the same number of rows!")

        return Matrix([
            [ x for x in self.matrix[row] ] + [ x for x in other.matrix[row] ] for row, _ in enumerate(self.matrix)
        ])
    
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


    m.append(m2).append(m2).print()