from matrix import Matrix
from vector import Vector

DATA_COUNT = 300
t = Vector([4, 3])

def randData():
	return Vector.randint(len(t)-1, 3).prepend([1])

X = Matrix([randData() for _ in range(DATA_COUNT)])
print(X)

y_hat = Matrix.fromVector(Vector([ x.dot(t) for x in X ]) + Vector.randfloat(DATA_COUNT, 1) - 0.5)
print(y_hat)

theta_best = ((X.transpose() * X).inverse() * X.transpose() * y_hat).toVector()
print(theta_best)

newX = randData()
y_predicted = newX.dot(theta_best)
print(newX, y_predicted)