#!/usr/bin/env python
######################################
# User: Jiang Li
# Date: Sat Jun  9 14:42:35 PDT 2018
######################################

## 1. Load package for the assignment
import numpy as np

## 2. Create matrix A with size (3,5) containing random numbers A = np.random.random(15)
print("## 2. Create matrix A with size (3,5) containing random numbers A = np.random.random(15)")

np.random.seed(123) 
A = np.matrix(np.random.random((3,5)))
print("A:")
print(A)

## 3.  Find the size and length of matrix A
print("## 3.  Find the size and length of matrix A")

print("The Size of A: ", A.shape)
print("The length of A:",A.size)

## 4.  Resize (crop/slice) matrix A to size (3,4)
print("## 4.  Resize (crop/slice) matrix A to size (3,4)")

A.resize(3,4,refcheck=False)
print("A now:")
print(A)

## 5.  Find the transpose of matrix A and assign it to B
print("## 5.  Find the transpose of matrix A and assign it to B")

B = A.T
print("A's transpose:")
print(B)

## 6.  Find the minimum value in column 1 of matrix B (check the properties of a matrix  B.min())
print("## 6.  Find the minimum value in column 1 of matrix B (check the properties of a matrix  B.min())")

min_1st_column = B[:,0].min()
print("Minimum value in column 1 of matrix B:",min_1st_column)
print("Minimum value in matrix B:",B.min())


## 7.  Find the minimum and maximum values for the entire matrix A
print("## 7.  Find the minimum and maximum values for the entire matrix A")

min_A = A.min()
max_A = A.max()
print("Minimum of A:",min_A)
print("Maximum of A:",max_A)

## 8.  Create vector X (an array) with 4 random numbers
print("## 8.  Create vector X (an array) with 4 random numbers")

np.random.seed(321)
X = np.array(np.random.random(4))
print("X:")
print(X)

## 9.  Create a function and pass vector X and matrix A in it
print("## 9.  Create a function and pass vector X and matrix A in it")


## 10.  In the new function multiply vector X with matrix A and assign the result to D (note: you may get an error! ... think why and fix it. Recall matric manipula4on in class!)
print("## 10.  In the new function multiply vector X with matrix A and assign the result to D (note: you may get an error! ... think why and fix it. Recall matric manipulation in class!)")

def mul(X,A):
    try:
        return X*A
    except ValueError as e:
        print(e)

D = mul(X,A.reshape(4,3))
print("muliply X and A (mul(X,A))",mul(X,A))
print("muliply X and A correctly (mul(X,A.reshape(4,3))",D)

## 11.  Create a complex number Z with absolute and real parts != 0
print("## 11.  Create a complex number Z with absolute and real parts != 0")

Z = np.array([1+9j,2+8j,3+7j],dtype=complex)
print("Z:",Z)

## 12.  Show its real and imaginary parts as well as it's absolute value
print("## 12.  Show its real and imaginary parts as well as it's absolute value")

real_v = Z.real
imaginary_v = Z.imag
absolute_v = np.absolute(Z)
print("Real value of Z:",real_v)
print("Imaginary value of Z:",imaginary_v)
print("Absolute value of Z:",absolute_v)


## 13.  Multiply result D with the absolute value of Z and record it to C
print("## 13.  Multiply result D with the absolute value of Z and record it to C")

C = D * absolute_v.reshape(3,1)
print("D multiply absolute value of Z:", C)

## 14.  Convert matrix B from a matrix to a string and overwrite B
print("## 14.  Convert matrix B from a matrix to a string and overwrite B")
print("Before converting, B:",B)
B = np.array( [str(i) for i in B.flatten().tolist()[0] ],dtype="string").reshape(4,3)
print("Before converting, B:",B)

## 15. Display a text on the screen: 'Your Name is done with HW2'
print("## 15. Display a text on the screen: 'Your Name is done with HW2'")

print("Jiang Li is done with HW2")


