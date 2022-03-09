class Matrix:
    def __init__(self, default):
        if not isinstance(default, list) or len(default) == 0 or not isinstance(default[0], list) or len(default[0]) == 0:
            raise ValueError("Invalid matrix!")

        self.matrix = default
    
    def append(self, other):
        if len(self.matrix) != len(other.matrix):
            raise ValueError("Matrices must have the same number of rows!")

        return Matrix([
            self.matrix[row] + other.matrix[row] for row, _ in enumerate(self.matrix)
        ])
    
    def print(self):
        """Prints the matrix.
    
        :returns: void
        """

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