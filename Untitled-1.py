
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
def Median_filter(arr,size):
    filter = np.ones(size)
    for i in range(1,len(arr)):
        for k in range(0,size):
            arr[i] += arr[i-k]*filter[k]

# def Medfilt (x, k):
#     k2 = (k - 1) // 2
#     y = np.zeros((len (x), k))
#     y[:,k2] = x
#     for i in range (k2):
#         j = k2 - i
#         y[j:,i] = x[:-j]
#         y[:j,i] = x[0]
#         y[:-j,-(i+1)] = x[j:]
#         y[-j:,-(i+1)] = x[-1]
#     return np.median (y, axis=1)

arr = [1,4,6,8,0,9,4,9,9,123]
arr1 = [1,20,40,60,0,40,60,80,123]
# Medfilt(arr,3)
arr1 = medfilt(arr1,3)


print(arr)
print(arr1)
plt.subplot(2,1,1)
plt.stem(arr)

plt.subplot(2,1,2)
plt.stem(arr1)
plt.show()
