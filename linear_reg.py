from __future__ import annotations
from typing import Tuple

from matrix import Matrix
from vector import Vector
from number import number

class LinearRegression:

	def __init__(self, feature_values: list[Vector], actual_values: Vector):
		X = Matrix(feature_values)
		y = Matrix.fromVector(actual_values)

		# fit the data
		self.feature_weights = ((X.T * X).inv * X.T * y).toVector()
	
	def weights(self):
		return self.feature_weights

	def predict(self, feature_values: Vector | list[number]):
		expected = len(self.feature_weights)-1
		if len(feature_values) != expected:
			raise ValueError(f"You must provide exactly {expected} feature values! Provided: {len(feature_values)}")
		
		if isinstance(feature_values, list):
			feature_values = Vector(feature_values)

		return feature_values.prepend([1]).dot(self.feature_weights)
