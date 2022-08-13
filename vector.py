from __future__ import annotations
import random
import math

from number import number

class Vector:
	def __init__(self, data: list[number]):
		"""Instantiate a Vector with some data."""
		
		self._inner_list = data
	
	@classmethod
	def rpt(cls, value: number, length: int):
		"""Instantiate a Vector with a repeating value."""
		
		return Vector([ value for _ in range(length) ])

	@classmethod
	def zeroes(cls, length: int):
		"""Instantiate a Vector with zeroes."""
		
		return Vector.rpt(0, length)

	@classmethod
	def ones(cls, length: int):
		"""Instantiate a Vector with ones."""
		
		return Vector.rpt(1, length)

	@classmethod
	def randint(cls, length: int, max: int):
		"""Instantiate a Vector with random ints, in the interval [0, max) - excluding max."""
		
		return Vector([ random.randint(0, max-1) for _ in range(length)])

	@classmethod
	def randfloat(cls, length: int, max: float):
		"""Instantiate a Vector with random floats, in the interval [0, max) - excluding max."""
		
		return Vector([ random.random()*max for _ in range(length)])

	def dot(self, other: Vector) -> number:
		"""Get the dot product of this vector and another."""

		if len(self) != len(other):
			raise ValueError("Vectors must have same length.")

		return sum([self[i] * other[i] for i in range(len(self))])

	def addVector(self, other: Vector):
		"""Add this vector to another, element-wise."""

		if len(self) != len(other):
			raise ValueError("Vectors must have same length.")
		
		return Vector([self[i] + other[i] for i in range(len(self))])
	
	def multiplyScalar(self, scalar: number):
		"""Multiply each element in this vector by a scalar."""

		return Vector([x * scalar for x in self])

	def addScalar(self, scalar: number):
		"""Add each element in this vector to a scalar."""

		return Vector([x + scalar for x in self])

	def prepend(self, other: Vector | list[number]):
		"""Append a vector's elements to the start of another."""

		return Vector(other._inner_list if isinstance(other, Vector) else other + self._inner_list)

	def append(self, other: Vector | list[number]):
		"""Append a vector's elements to the end of another."""

		return Vector(self._inner_list + other._inner_list if isinstance(other, Vector) else other)

	def copy(self):
		return Vector([x for x in self])

	def toList(self):
		return [x for x in self._inner_list]

	# Overloading
	def __len__(self):
		return len(self._inner_list)

	def __getitem__(self, index: int):
		return self._inner_list[index]

	def __setitem__(self, index: int, value: number):
		self._inner_list[index] = value

	def __delitem__(self, index: int):
		del self._inner_list[index]

	def __add__(self, other: Vector | number):
		return self.addVector(other) if isinstance(other, Vector) else self.addScalar(other)

	def __sub__(self, other: Vector | number):
		return self + -other

	def __mul__(self, other: number):
		return self.multiplyScalar(other)

	def __pos__(self):
		return self.copy()
	
	def __neg__(self):
		return self * -1

	def __eq__(self, other: Vector):
		if len(self) != len(other):
			return False

		for i in range(len(self)):
			if not math.isclose(self[i], other[i]):
				return False

		return True

	def __iter__(self):
		return VectorIter(self)

	def __str__(self):
		return f"[{', '.join([str(x) for x in self])}]"

	def __repr__(self):
		return str(self)

class VectorIter:
	def __init__(self, vector: Vector):
		self._v = vector
		self._index = 0

	def __iter__(self): return self

	def __next__(self):
		if self._index < len(self._v):
			ret = self._v[self._index]
			self._index += 1
			return ret
			
		raise StopIteration

if __name__ == "__main__":
	for x, y in enumerate(Vector.randfloat(10, 5)):
		print(x, y)