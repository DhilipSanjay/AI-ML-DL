# Numpy Basics

**Date:** 25, March, 2021

**Author:** Dhilip Sanjay S

---

## [Python File - Numpy Basics](https://github.com/DhilipSanjay/AI-ML-DL/blob/main/Numpy/numpy-basics.py)

```py
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
multi[:, 2] = 10 #3nd col
print(multi)
print("\nChanging a row value")
multi[1, :] = 8  #1st row
print(multi)
print("\nChanging a subsequence of same size")
multi[:, 4] = [1, 2]
print(multi)

# ---------------------------------

# 3-Dimensional array
print("\n3 Dimensional Array")
d3 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
print(d3)

# Get specific element (work outside in)
print("\n3D - Get specific element")
print(d3[0, 1, 1])
print(d3[0, 1, :])
print(d3[0, :, 0])
print(d3[:, 1, 1])
print(d3[:, :, 1])
print(d3[1, :, :])

print("\n3D - Replace element")
print(d3)
d3[:, 0, :] =[[0, 0], [1, 1]]
print(d3)
# ---------------------------------

#  Initializing arrays
print("\nInitializing arrays")
# Note that the arguments are tuples
print("\nAll zeros")
print(np.zeros(5))
print(np.zeros((2, 3, 4)))

print("\nAll ones")
print(np.zeros(5))
print(np.zeros((4, 3, 4), dtype='float16'))

print("\nAny number - using full")
print(np.full((2,3), 3, dtype='float32'))
# Also check out order = 'C' or 'F' - Row or Column wise

print("\nAny number - using full like")
# full_like -> uses the same data type of another array
# So, we have to change it using dtype
print(np.full_like(multi, 99, dtype='int8'))

# using full -> give only shape of another array
print(np.full(multi.shape, 99))

# ---------------------------------

# Random numbers
print("\nRandom Numbers")
# Note that the argument is not a tuple
print("\nRandom Numbers - using rand")
print(np.random.rand(4,3)) 

print("\nRandom Numbers - using random_sample")
print(np.random.random_sample(multi.shape))

print("\nRandom Integers - using rand")
# Start, End, size
print(np.random.randint(1, 10, (4,4)))

# ---------------------------------

print("\nIdentity matrix")
identity = np.identity(4)
print(identity)

# ---------------------------------

print("\nRepeat the array")
print(np.repeat(a, 2))

print("\nRepeat the array - down the rows")
print(np.repeat(multi, 2, axis = 0))

print("\nRepeat the array - across the columns")
print(np.repeat(multi, 2, axis = 1))

print("\nRepeat an one dimensional array - using expand dim and repeat")
print(a)
a_2d = np.expand_dims(a, axis=0)
print("\nUsing axis = 0 :\n", a_2d)
print(np.repeat(a_2d, 4, axis=0))
a_2d = np.expand_dims(a, axis=1)
print("\nUsing axis = 1 :\n", a_2d)
print(np.repeat(a_2d, 4, axis=1))

# ---------------------------------

print("\nEvenly spaced values within a given interval")

print("\n Using arange - for integer step size")
# For non-integer step size, the result may be inconsistent - so use linspace
print(np.arange(1, 10, 2))

print("\n Using Linspace")
print(np.linspace(1, 10, 20, endpoint=False, retstep=True))
print(np.linspace(1, 10, 20, endpoint=True, retstep=True))

# ---------------------------------

np_1s = np.ones((2,4))
np_9s = np.full((2,4), 9)

print("\nConcatenation")
print(np.concatenate([np_1s, np_9s], axis=0))
print(np.concatenate([np_1s, np_9s], axis=1))

# ---------------------------------

print("\nSum")
print(np.sum(np_9s, axis=0))
print(np.sum(np_9s, axis=1))

# ---------------------------------

#  Item wise computation
print("\nItem wise computation")
print(a*b)
print(a+2)
print(a/2)
print(a**2)
print(np.sin(a))

# ---------------------------------

# Exercise
print("\nEXERCISE")
'''
1 1 1 1 1
1 0 0 0 1
1 0 9 0 1
1 0 0 0 1
1 1 1 1 1
'''
ex1 = np.ones((5,5))
print(ex1)
ex1[1:4, 1:4] = np.zeros((3,3))
print(ex1)
ex1[2, 2] = 9
print("\nSolution 1: \n", ex1)

# ---------------------------------

# Copying an array
print("\nCopying")
copy_a = a.copy()
print(copy_a)
copy_a[1] = 1000
print(copy_a)
print(a)

# ---------------------------------

# Linear Algebra
print("\nLinear Algebra functions")
a = np.ones((2, 3))
b = np.full((3, 2), 2)
print(a,b)

print("\n Matrix multiplication")
print(np.matmul(a, b))

print("\nDeterminant")
print(np.linalg.det(identity))

# ---------------------------------

# Statistics
print("\nStatistics functions")
stats = np.array([[1,2,3], [4,5,6]])
print(np.min(stats))
print(np.max(stats, axis=0))
print(np.max(stats, axis=1))

print(np.sum(stats, axis=0))
print(np.sum(stats, axis=1))

# ---------------------------------

print("\nReorganizing Arrays")
print(stats)
print(stats.reshape((6, 1)))
print(stats.reshape((3, 2)))
print(stats.reshape((1, 6)))

print("\nVertical Stack")
print(np.vstack((stats, stats)))

print("\nHorizontal Stack")
print(np.hstack((stats, stats)))

# ---------------------------------

print("Load Data from file")
data = np.genfromtxt('data.txt', delimiter=',')
print(data) # Defaults to float
dataint = data.astype('int32') # Make a int copy
print(dataint)

# ---------------------------------

print("\nBoolean Masking")
print(dataint > 50)

print("\nnp.any function")
print(np.any(dataint > 50, axis=0))
print("\nnp.all function")
print(np.all(dataint > 50, axis=0))
print("\n& in condition")
print(((dataint > 50) & (dataint < 100)))
print("\n~ and & in condition")
print(~((dataint > 50) & (dataint < 100)))

print("\nAdvanced indexing")
print(dataint[dataint > 50])

b = np.array([1,2,3,4,5,6,7,8,9,0])
print(b[[1, 2, 5]])

# ---------------------------------

# Exercise
print("\nEXERCISE 2")
ex2 = np.arange(1, 31, 1).reshape((6,5))
print(ex2)
'''
Index:
11 12
16 17
'''
print(ex2[2:4, 0:2])

'''
Index:
2
  8
    14
       20
'''
print(ex2[[0,1,2,3], [1,2,3,4]])

'''
Index:
 4  5
24 25
29 30
'''
print(ex2[(0, 4, 5), 3:])

# ---------------------------------
```
