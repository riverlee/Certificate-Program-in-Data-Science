#####Information section#########
## Name:  Jiang Li
## Email: riverlee2008@gmail.com
###############################
import numpy as np
import matplotlib.pyplot as plt

## Ab array A with 12 elemetns beginning from number 5, containing consective odd numbers.

A = np.linspace(start = 5, stop = 27, num = 12,dtype=np.float)


## Compute the simple moving average of an array(SMA), using an adjustable window using a function, the function calculating the SMA of the array must take tow input arguments: the array(A), a window width with a default value =2

def SMA(A,width=2):
    ## last index
    last_index = len(A)-width+1
    ## Use list comprehension and return as numpy
    return np.array([np.mean(A[i:(i+width)]) for i in range(last_index)],dtype=A.dtype)
    
    
## After every SMA is collected(as array), calculate its current cumulative moving average(CMA)

def CMA(B):
    ## get lens
    lens = [i+1 for i in range(len(B))] 
    return np.cumsum(B)/lens
    
    
width=2
B = SMA(A,width=width)
C = CMA(B)

## Print the result from the two calls made above using a short description of the data:
print("The original array is", A)
print("Current window width=",width)
print("The SMA result is :",B)
print("The CMS of this SMA is: ",C)


## plot:
### The sin function for array B and C, specifying different color,linewidth, lable and figure name passed to string to hte ploting function
### Show the grid, lengend
plt.figure("Midterm")
plt.plot(np.sin(B),linewidth=1.5,linestyle="-",label="SMA",color='blue')

plt.plot(np.sin(C),linewidth=2.5,linestyle="-.",label="CMA",color='red')

plt.title("Midterm - Jiang")
plt.grid()
plt.legend(loc='upper right')
plt.show()
