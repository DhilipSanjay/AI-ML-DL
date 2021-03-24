# Introduction to Numpy

**Date:** 24, March, 2021

**Author:** Dhilip Sanjay S

---

## What is Numpy?
- NumPy is the fundamental package for scientific computing in Python. 
- It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays

---

## Lists vs Numpy Arrays
### Lists
- Slow 
- Fixed size (Int32, Int16, Int8)
- Type checking is necessary (it can store any data type)
- Non contiguous memory

### Numpy Arrays
- Fast
- Size, Reference count, Object Type and value - everything will be stored 
- No type checking is required (homogeneous elements)
- Contiguous memory - Benefits:
    - SIMD Vector Processing
    - Effective cache utilization
- Applications
    - MATLAB replacement
    - Plotting (Matplotlib)
    - Backend (Pandas, Store images)
    - Machine Learning

---

### Reference Links
- [Numpy Documentation](https://numpy.org/doc/stable/user/quickstart.html)

---