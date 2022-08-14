from __future__ import annotations
from typing import Tuple

from vector import Vector
from number import number

class Matrix:
	def __init__(self, default: list[list[number]] | list[Vector]):
		self.matrix: list[Vector] = [x if isinstance(x, Vector) else Vector(x) for x in default]

	@classmethod
	def fromVector(cls, vector: list[number] | Vector):
		"""Instantiates a new Matrix using a vector as a single column."""

		return Matrix([ [x] for x in vector ])

	@classmethod
	def zero(cls, rows: int, cols: int):
		"""Instantiates new Matrix filled with zeroes of size rows x cols."""

		return Matrix([
			Vector.zeroes(cols) for _ in range(rows)
		])
	
	@classmethod
	def identity(cls, size):
		"""Instantiates a new identity matrix with the given size.
		
		I_size.
		"""

		return Matrix([
			[ 1 if r == c else 0 for c in range(size) ] for r in range(size)
		])

	def add(self, other: Matrix) -> Matrix:
		"""Sums two matrices cell-wise. That is, return a matrix where each cell (C)ij is equal to (A)ij + (B)ij."""

		if not self.sameSizeAs(other):
			raise ValueError(f"Matrices must be the same size to add. Left: {self.getSize()}, Right: {other.getSize()}")
		
		return Matrix([ 
			self[row] + other[row]
			for row in range(len(self)) 
		])

	def scalarMultiply(self, k: number) -> Matrix:
		"""Multiplies the matrix by a constant, where each cell (B)ij = (A)ij * k."""
		return Matrix([ row*k for row in self.matrix ])

	def append(self, other: Matrix) -> Matrix:
		"""Appends one matrix onto the right of another.

		Rows must match in size. Does not modify the original matrix in-place.

		Returns a new matrix.
		"""

		if len(self) != len(other):
			raise ValueError("Matrices must have the same number of rows!")

		return Matrix([
			self[row].append(other[row]) for row in range(len(self))
		])
	
	def multiplyByMatrix(self, other: Matrix) -> Matrix:
		"""Multiplies two matrices.
		
		Returns a new matrix.
		Note that the number of columns on the left matrix must equal the number of rows on the right matrix.
		"""

		if len(self[0]) != len(other):
			raise ValueError(f"The number of columns on the left matrix must equal the number of rows on the right matrix. Left columns: {len(self[0])}, Right rows: {len(other)}")

		n = len(self)
		k = len(other[0])

		# initialise new matrix with zeroes
		result = Matrix.zero(n, k)
	
		for row in range(n):
			for col in range(k):
				result[row][col] = self[row].dot(other.getCol(col))

		return result

	def modulo(self, mod: number) -> Matrix:
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
		newM[rowI] = self[rowJ]
		newM[rowJ] = self[rowI]
		
		return newM

	def swap(self, rowI: int, rowJ: int) -> Matrix: 
		"""Swaps two rows. Returns a new matrix. Alias of Matrix#interchange()."""
		return self.interchange(rowI, rowJ)

	def multiplyRow(self, rowI: int, k: number) -> Matrix:
		"""Multiply row by constant k.
		
		Ri = k * Ri

		Returns a new matrix.
		"""

		newM = self.copy()
		newM[rowI] = newM[rowI]*k

		return newM

	def addMultipleOfRow(self, rowI, rowJ, k):
		"""Add k times rowJ to rowI.
		
		Ri = Ri + k*Rj

		Returns a new matrix.
		"""

		newM = self.copy()
		newM[rowI] = self[rowI] + self[rowJ] * k

		return newM

	def toRowEchelon(self, maxCol=-1) -> Matrix:
		"""Uses Gaussian Elimination to reduce the matrix to row echelon form.
		
		Returns a new matrix.

		maxCol: the maximum column to reduce to row-echelon form with. All columns get transformed with a row, but columns past maxCol won't be affected. Inclusive.
		"""

		if maxCol == -1: maxCol = len(self[0])-1

		newM = self.copy()
		row = 0
		col = 0

		while row < len(newM) and col < len(newM[0]) and col <= maxCol:
			# Step 1: Locate the leftmost column that does not consist entirely of zeroes
			c = newM.getCol(col)
			nonzero = -1

			for i, cell in enumerate(c):
				if i < row:
					continue
				
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

			print("1", nonzero, row, col, newM)

			# Step 3: If the entry that is now at the top of the column in step is a, multiply the first row by 1/a in order to introduce a leading 1.
			newM = newM.multiplyRow(row, 1 / newM[row][col])

			print("2", row, col, newM)

			# Step 4: Add suitable multiples of the top row to the rows below so that all entries below the leading 1 become zeroes
			for i in range(row+1, len(self)):
				k = -newM[i][col] / newM[row][col]
				newM = newM.addMultipleOfRow(i, row, k)
				print("3", i, row, col, newM)

			# Step 5: Now cover the top row in the matrix and begin again with Step 1 applied to the submatrix that remains. 
			row += 1
			col += 1 # advance column since we already worked with the current column - we know it's all zeroes for the rows we're checking on the next iteration

		return newM

	def toReducedRowEchelon(self, maxCol = -1) -> Matrix:
		# NOTE: currently assumes valid input

		newM = self.copy().toRowEchelon(maxCol)

		if maxCol == -1:
			maxCol = len(newM[0])-1

		# Step 6. Beginning with the last nonzero row and working upwards, add
		# suitable multiples of each row to the rows above to introduce zeros
		# above the leading 1’s.

		row = len(newM)-1

		while row >= 0:
			# Get leading column
			leading = -1
			for col in range(maxCol+1):
				if newM[row][col] != 0:
					leading = col
					break

			# All zeroes, move onto next row
			if leading == -1:
				row -= 1
				continue

			for i in range(row-1, -1, -1):
				newM = newM.addMultipleOfRow(i, row, -newM[i][leading]/newM[row][leading])

			row -= 1

		return newM

	def transpose(self) -> Matrix:
		"""Transposes the matrix. That is, it returns B where (B)ji = (A)ij. Or, B = A^T."""
		rows, cols = self.getSize()

		return Matrix([
			self.getCol(col) for col in range(cols)
		])

	@property
	def T(self) -> Matrix: return self.transpose()

	def isInverse(self, other: Matrix) -> bool:
		"""Returns true if the matrices are inverses of each other."""

		return (
			self.isSquare() and 
			self.sameSizeAs(other) and
			self * other == Matrix.identity(len(self)) and
			other * self == Matrix.identity(len(self))
		)

	def inverseOf2by2(self) -> Matrix:
		"""Finds the inverse of a 2x2 matrix using the simple formula."""

		if not self.isSquare() or not self.getSize()[0] == 2:
			raise ValueError(f"Matrix must be 2x2! Size: {self.getSize()}")
		
		a = self[0][0]
		b = self[0][1]
		c = self[1][0]
		d = self[1][1]

		k = 1 / (a*d-b*c)

		newM = Matrix([
			[d, -b],
			[-c, a]
		]) * k

		return newM

	def exceptRow(self, rowIndex: int) -> Matrix:
		"""Returns the given matrix except the specified row index."""

		return Matrix([ x for i, x in enumerate(self) if i != rowIndex ])
	
	def exceptColumn(self, colIndex: int) -> Matrix:
		"""Returns the given matrix except the specified row index."""

		res = self.copy()
		for i in range(len(self)):
			del res[i][colIndex]

		return res
	
	def minorOf(self, row: int, col: int):
		"""Finds the minor of the matrix at M(row, col)."""

		if not self.isSquare() and len(self) > 1:
			raise ValueError(f"Matrix must be a square and at least 2x2! Size: {self.getSize()}")

		return self.exceptRow(row).exceptColumn(col).determinant()
	
	def cofactorOf(self, row: int, col: int):
		"""Finds the cofactor of the matrix at M(row, col)."""

		if not self.isSquare() and len(self) > 1:
			raise ValueError(f"Matrix must be a square and at least 2x2! Size: {self.getSize()}")

		sign = (1 if (row + col) % 2 == 0 else -1)
		
		return sign * self.minorOf(row, col)

	def determinant(self) -> number:
		"""Finds the determinant of the matrix, if the matrix is square."""

		if not self.isSquare():
			raise ValueError(f"Matrix must be a square! Size: {self.getSize()}")

		if self.getSize()[0] == 1: return self[0][0]

		result = 0
		size = len(self)
		
		for i in range(size):
			result += self[0][i] * self.cofactorOf(0, i)

		return result

	def adjoint(self):
		"""Constructs the corresponding adjoint matrix."""

		return Matrix([ [ self.cofactorOf(row, col) for col in range(len(self[row])) ] for row in range(len(self)) ]).transpose()

	def hasInverse(self):
		"""Checks the determinant of the matrix to determine if it is invertible."""
		
		return not self.isSquare() and self.determinant() != 0

	def inverse(self):
		"""Finds the inverse of the matrix, if it exists."""

		if not self.isSquare():
			raise ValueError(f"Matrix must be square! Size: {self.getSize()}")

		det = self.determinant()

		if det == 0:
			raise ValueError(f"Matrix must not have a determinant of zero (i.e. be invertible).")

		return self.adjoint() / det

	@property
	def inv(self): return self.inverse()

	def isDiagonalMatrix(self) -> bool:
		for rowI, row in enumerate(self.matrix):
			for colI, cell in enumerate(row):
				if rowI != colI and cell != 0:
					return False

		return True 

	def isSquare(self) -> bool:
		"""Returns true if the matrix is a square, defined as having an equal number of rows and columns. That is, it's size can be defined as n x n."""
		rows, cols = self.getSize()

		return rows == cols
	
	def isSymmetric(self) -> bool:
		if not self.isSquare(): 
			raise ValueError(f"Matrix must be square. Size: {self.getSize()}")

		rows, cols = self.getSize()

		for row in range(rows):
			for col in range(row+1, cols):
				if self[row][col] != self[col][row]:
					return False
		
		return True

	def isSkewSymmetric(self) -> bool:
		if not self.isSquare(): 
			raise ValueError(f"Matrix must be square. Size: {self.getSize()}")

		rows, cols = self.getSize()

		for row in range(rows):
			for col in range(row+1, cols):
				if self[row][col] != -self[col][row]:
					return False
		
		return True

	def sameSizeAs(self, other: Matrix) -> bool:
		"""Returns true if the given matrix has an equal number of rows and columns as the matrix."""

		selfRows, selfCols = self.getSize()
		otherRows, otherCols = other.getSize()

		return selfRows == otherRows and selfCols == otherCols

	def isEqualTo(self, other: Matrix) -> bool:
		"""Returns true if the two matrices are equivalent.
		
		That is, they have the same number of rows and columns, and each element is close to the corresponding element in the other matrix. Comparison of elements is done with math.iscloseto, to account for floating point bugs.
		"""

		if not self.sameSizeAs(other):
			return False

		for row in range(len(self)):
			if self[row] != other[row]:
				return False

		return True

	def toVector(self):
		"""Converts this matrix to a Vector."""

		_, cols = self.getSize()
		if cols != 1:
			raise ValueError("Matrix must be a column matrix.")
		
		return self.getCol(0)

	def getCol(self, i: int) -> Vector:
		"""Returns a column of the matrix as a Vector."""

		return Vector([ row[i] for row in self.matrix ])

	def getRow(self, i: int) -> Vector:
		"""Returns a row of the matrix as a Vector."""
		return self.matrix[i]

	def getSize(self) -> Tuple[int, int]:
		"""Returns a tuple defining the size of the matrix in the form [rows, cols]."""
		return len(self), len(self[0])

	def copy(self):
		"""Returns a deep copy of the matrix."""
		
		copy = Matrix([
			row.copy() for row in self.matrix
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

	def __mul__(self, other: Matrix | number) -> Matrix: 
		if isinstance(other, number):
			return self.scalarMultiply(other)
		elif isinstance(other, Matrix):
			return self.multiplyByMatrix(other)

		raise TypeError(f"Invalid matrix multiplication operand: {type(other)}")
	
	def __truediv__(self, other: float) -> Matrix:
		return self * (1 / other)
	
	def __pos__(self) -> Matrix: return self

	def __neg__(self) -> Matrix:
		return self * -1

	def __eq__(self, other) -> bool: return self.isEqualTo(other)

	def __getitem__(self, rowIndex: int):
		return self.getRow(rowIndex)
	
	def __setitem__(self, rowIndex: int, value: list[number] | Vector):
		if isinstance(value, Vector):
			self.matrix[rowIndex] = value
			return

		self.matrix[rowIndex] = Vector(value)

	def __len__(self):
		return len(self.matrix)

	def __iter__(self):
		return MatrixIter(self)

	def __str__(self):
		o = ""

		for row in self.matrix:
			o += "["
			for cell in row:
				o += f" {cell} "
			o += "]\n"
		
		return o

class MatrixIter:
	def __init__(self, matrix: Matrix):
		self._m = matrix
		self._index = 0

	def __iter__(self): return self

	def __next__(self):
		if self._index < len(self._m):
			ret = self._m[self._index]
			self._index += 1
			return ret
			
		raise StopIteration

if __name__ == "__main__":
	m = Matrix([
		[2, 7, -2, 2],
		[4, 14, -4, 4],
		[-3, -6, 1, -2],
		[-6, -3, -2, -2]
	])

	n = Matrix([
		[1, -2, 1],
		[-1, 2, 2],
		[2, 3, -2]
	])

	print(n.inverse() * -21)
