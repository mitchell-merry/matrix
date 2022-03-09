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

        if len(self) != len(other):
            raise ValueError("Matrices must have the same number of rows!")

        return Matrix([
            self.copyRowByIndex(row) + other.copyRowByIndex(row) for row, _ in enumerate(self.matrix)
        ])
    
    def multiply(self, other: Matrix) -> Matrix:
        """Multiplies two matrices.
        
        Returns a new matrix.
        Note that the number of columns on the left matrix must equal the number of rows on the right matrix.
        """

        if len(self[0]) != len(other):
            raise ValueError("The number of columns on the left matrix must equal the number of rows on the right matrix.")

        n = len(self)
        k = len(other[0])

        # initialise new matrix with zeroes
        result = [[0 for __ in range(k)] for _ in range(n)]
    
        for row in range(n):
            for col in range(k):
                result[row][col] = dot_product(self[row], other.getCol(col))

        return Matrix(result)

    def interchange(self, rowI: int, rowJ: int) -> Matrix:
        """Swaps two rows. Returns a new matrix."""

        newM = self.copy()
        newM[rowI] = self.copyRowByIndex(rowJ)
        newM[rowJ] = self.copyRowByIndex(rowI)
        
        return newM

    def swap(self, rowI: int, rowJ: int) -> Matrix: 
        """Swaps two rows. Returns a new matrix. Alias of Matrix#interchange()."""
        return self.interchange(rowI, rowJ)

    def multiplyRow(self, rowI: int, k: int) -> Matrix:
        """Multiply row by constant k.
        
        Ri = k * Ri

        Returns a new matrix.
        """

        newM = self.copy()
        newM[rowI] = multiplyList(newM[rowI], k)

        return newM

    def addMultipleOfRow(self, rowI, rowJ, k):
        """Add k times rowJ to rowI.
        
        Ri = Ri + k*Rj

        Returns a new matrix.
        """

        newM = self.copy()
        newM[rowI] = addLists(self[rowI], multiplyList(self[rowJ], k))

        return newM

    def getCol(self, i: int) -> list[int]:
        """Returns a column of the matrix as a list of numbers."""

        return [ row[i] for row in self.matrix ]

    def copyRow(self, row: list[int]) -> list[int]:
        return [ x for x in row ]

    def copyRowByIndex(self, i: int) -> list[int]:
        return [ x for x in self[i] ]

    def copy(self):
        """Returns a deep copy of the matrix."""
        
        copy = Matrix([
            [ x for x in row ] for row in self.matrix
        ])

        return copy

    def print(self):
        """Prints the matrix & returns it."""

        print(self)
        return self
        

    # Overloading.
    def __add__(self, other: Matrix): return self.append(other)
    def __mul__(self, other: Matrix): return self.multiply(other)

    def __getitem__(self, rowIndex: int) -> list[int]:
        return self.matrix[rowIndex]
    
    def __setitem__(self, rowIndex: int, value: list[int]):
        self.matrix[rowIndex] = value

    def __len__(self):
        return len(self.matrix)

    def __str__(self):
        o = ""

        for row in self.matrix:
            o += "["
            for cell in row:
                o += f" {cell} "
            o += "]\n"
        
        return o

# dot product of [a, b, c] and [d, e, f] is (a*d + b*e + c*f)
def dot_product(left: list[int], right: list[int]) -> int:
    """Calculates the dot product of two integer lists.
    
    Dot product of [a, b, c] and [i, j, k] is a*i + b*j + c*k.
    """

    if len(left) != len(right):
        raise ValueError("Length of two lists must be equal for dot product.")

    sum = 0
    for i in range(len(left)):
        sum += left[i] * right[i]
    return sum

def addLists(left: list[int], right: list[int]) -> list[int]:
    """Adds two lists.
    
    Addition of [a, b, c] and [i, j, k] is [a+i, b+j, c+k].
    """

    if len(left) != len(right):
        raise ValueError("Length of two lists must be equal for addition.")

    return [ left[i] + right[i] for i in range(len(left)) ]

def multiplyList(arr: list[int], k: int):
    """Multiplies a list of numbers by a constant.
    
    Multiplication of [1, 2, 3] by 4 is [4, 8, 12].
    """
    
    return [ k*value for value in arr ]

if __name__ == "__main__":
    m = Matrix([
        [0, -3],
        [1, 2],
        [9, 4]
    ])

    m2 = Matrix([
        [5, -3, 2],
        [7, 0, 5]
    ])

    print(m.addMultipleOfRow(0, 1, 3))