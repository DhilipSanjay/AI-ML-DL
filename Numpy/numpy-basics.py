import numpy as np

# Numpy array along with dtype
a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([2, 3, 4, 7, 8, 9], dtype='int16')

# Multidimensional array
print("\nMultidimensional array ")
multi = np.array([a, b], dtype="float")
print(multi)

# Get the dimensions
print("\nDimensions")
print(a.ndim)
print(multi.ndim)

# Get the shape
print("\nShape")
print(a.shape)
print(multi.shape)

# Get type
print("\nType")
print(a.dtype)
print(b.dtype)
print(multi.dtype)

# Get item size - in bytes
print("\nItemsize - size of one item")
print(a.itemsize)
print(b.itemsize)
print(multi.itemsize)

# Get size of the np array
print("\nsize - number of items in the array ")
print(a.size)
print(multi.size)

# Get total size
print("\nnbytes - total size (itemsize * size of one item)")
print(a.nbytes)
print(multi.nbytes)

# Get specific element [r, c]
print("\nSpecific element")
print(multi[0, 2])
print(multi[1, -2])

# Get specific row
print("\nSpecific row")
print(multi[0, :])

# Get specific column
print("\nSpecific columns")
print(multi[:, 1])

# Get elements with stepsize
# start : end : stepsize
print("\nUsing Stepsize")
print(multi[0, 1:6:2])

# Changing values
print("\nChanging values")
a[2] = 10
print(a)
print("\nChanging a column value")
multi[:, 2] = 10
print(multi)
print("\nChanging a row value")
multi[1, :] = 8
print(multi)

# ---------------------------------

# 3-Dimensional array

#  Item wise computation
print("\nItem wise computation" , a*b)


