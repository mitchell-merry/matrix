from matrix import Matrix
from vector import Vector
from linear_reg import LinearRegression

DATA_COUNT = 50
t = Vector([4, 3, 6])

def randData():
	return Vector.randint(len(t)-1, 3).prepend([1])

X = [randData() for _ in range(DATA_COUNT)]
y = Vector([ x.dot(t) for x in X ]) + Vector.randfloat(DATA_COUNT, 1) - 0.5

lg = LinearRegression(X, y)

print(lg.weights())

data = Vector([2, 3])
print("Data:", data)
print("Actual:", data.prepend([1]).dot(t))
print("Predicted: ", lg.predict([2, 3]))