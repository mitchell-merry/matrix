from __future__ import annotations
from typing import Tuple, Union

class Matrix:
    def __init__(self, default: list[list[float]]):

        # Matrix must be a double array where rows and cols are > 0
        if not isinstance(default, list) or len(default) == 0 or not isinstance(default[0], list) or len(default[0]) == 0:
            raise ValueError("Invalid matrix!")

        self.matrix = default
    
    def add(self, other: Matrix) -> Matrix:
        if not self.sameSizeAs(other):
            raise ValueError(f"Matrices must be the same size to add. Left: {self.getSize()}, Right: {other.getSize()}")
        
        return Matrix([ 
            [ 
                self[rowIndex][colIndex] + other[rowIndex][colIndex] 
                for colIndex, cell in enumerate(row)
            ] 
            for rowIndex, row in enumerate(self.copy().matrix) 
        ])

    def scalarMultiply(self, k: float | int) -> Matrix:
        return Matrix([
            [ cell*k for cell in row ] for row in self.matrix
        ])

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
    
    def multiplyByMatrix(self, other: Matrix) -> Matrix:
        """Multiplies two matrices.
        
        Returns a new matrix.
        Note that the number of columns on the left matrix must equal the number of rows on the right matrix.
        """

        if len(self[0]) != len(other):
            raise ValueError("The number of columns on the left matrix must equal the number of rows on the right matrix.")

        n = len(self)
        k = len(other[0])

        # initialise new matrix with zeroes (0.1 for float)
        result = [[0.1 for __ in range(k)] for _ in range(n)]
    
        for row in range(n):
            for col in range(k):
                result[row][col] = dotProduct(self[row], other.getCol(col))

        return Matrix(result)

    def modulo(self, mod: float) -> Matrix:
        """Applies a modulo operation to each cell in the matrix. 
        
        Undefined behaviour for cells < 0.
        Returns a new matrix.
        """

        newM = self.copy()
        for y in range(len(newM)):
            for x in range(len(newM[y])):
                newM[y][x] %= mod
        return newM
        

    def interchange(self, rowI: int, rowJ: int) -> Matrix:
        """Swaps two rows. Returns a new matrix."""

        newM = self.copy()
        newM[rowI] = self.copyRowByIndex(rowJ)
        newM[rowJ] = self.copyRowByIndex(rowI)
        
        return newM

    def swap(self, rowI: int, rowJ: int) -> Matrix: 
        """Swaps two rows. Returns a new matrix. Alias of Matrix#interchange()."""
        return self.interchange(rowI, rowJ)

    def multiplyRow(self, rowI: int, k: float) -> Matrix:
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

    def toRowEchelon(self) -> Matrix:
        """Uses Gaussian Elimination to reduce the matrix to row echelon form.
        
        Returns a new matrix.
        """

        newM = self.copy()
        row = 0
        col = 0

        while row < len(newM) and col < len(newM[0]):
            # Step 1: Locate the leftmost column that does not consist entirely of zeroes
            c = newM.getCol(col)
            nonzero = -1

            for i, cell in enumerate(c):
                if cell != 0:
                    nonzero = i
                    break
            
            # if it's all zeroes, go to the next column
            if nonzero == -1:
                col += 1
                continue

            # Step 2: Interchange the top row with another, if necessary, to bring a nonzero entry to the top of the column found in Step 1.
            if nonzero != 0:
                newM = newM.interchange(row, nonzero)

            # Step 3: If the entry that is now at the top of the column in step is a, multiply the first row by 1/a in order to introduce a leading 1.
            newM = newM.multiplyRow(row, 1 / newM[row][col])

            # Step 4: Add suitable multiples of the top row to the rows below so that all entries below the leading 1 become zeroes
            for i in range(row+1, len(self)):
                k = -newM[i][col] / newM[row][col]
                newM = newM.addMultipleOfRow(i, row, k)

            # Step 5: Now cover the top row in the matrix and begin again with Step 1 applied to the submatrix that remains. 
            row += 1
            col += 1 # advance column since we already worked with the current column - we know it's all zeroes for the rows we're checking on the next iteration

        return newM

    def sameSizeAs(self, other: Matrix) -> bool:
        selfRows, selfCols = self.getSize()
        otherRows, otherCols = other.getSize()

        return selfRows == otherRows and selfCols == otherCols

    def getCol(self, i: int) -> list[float]:
        """Returns a column of the matrix as a list of numbers."""

        return [ row[i] for row in self.matrix ]

    def getSize(self) -> Tuple[int, int]:
        return len(self), len(self[0])

    def copyRow(self, row: list[float]) -> list[float]:
        return [ x for x in row ]

    def copyRowByIndex(self, i: int) -> list[float]:
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
    def __add__(self, other: Matrix) -> Matrix: 
        return self.add(other)
    
    def __sub__(self, other: Matrix) -> Matrix:
        return self + -other

    def __mul__(self, other: Matrix | float | int) -> Matrix: 
        if isinstance(other, float) or isinstance(other, int):
            return self.scalarMultiply(other)
        elif isinstance(other, Matrix):
            return self.multiplyByMatrix(other)

        raise TypeError(f"Invalid matrix multiplication operand: {type(other)}")
    
    def __truediv__(self, other: float) -> Matrix:
        return self * (1 / other)
    
    def __pos__(self) -> Matrix: return self

    def __neg__(self) -> Matrix:
        return self * -1

    def __getitem__(self, rowIndex: int) -> list[float]:
        return self.matrix[rowIndex]
    
    def __setitem__(self, rowIndex: int, value: list[float]):
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

def dotProduct(left: list[float], right: list[float]) -> float:
    """Calculates the dot product of two integer lists.
    
    Dot product of [a, b, c] and [i, j, k] is a*i + b*j + c*k.
    """

    if len(left) != len(right):
        raise ValueError("Length of two lists must be equal for dot product.")

    sum = 0
    for i in range(len(left)):
        sum += left[i] * right[i]
    return sum

def addLists(left: list[float], right: list[float]) -> list[float]:
    """Adds two lists.
    
    Addition of [a, b, c] and [i, j, k] is [a+i, b+j, c+k].
    """

    if len(left) != len(right):
        raise ValueError("Length of two lists must be equal for addition.")

    return [ left[i] + right[i] for i in range(len(left)) ]

def multiplyList(arr: list[float], k: float):
    """Multiplies a list of numbers by a constant.
    
    Multiplication of [1, 2, 3] by 4 is [4, 8, 12].
    """

    return [ k*value for value in arr ]

if __name__ == "__main__":
    A = Matrix([
        [1, -2, -1, -1],
        [3, 0, 6, 11]
    ])

    print(A * 3)
    print(-A)

    A = Matrix([
        [-2, 3],
        [1, 1]
    ])

    B = Matrix([
        [4, -1],
        [2, 2]
    ])

    print(B * 3 - A / 2)
    print(A * 0)