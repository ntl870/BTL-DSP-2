import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res


def CalculateAC(data):
    AC = np.zeros(len(data))
    for i in range(0, len(data)):
        for k in range(0, len(data) - i-1):
            AC[i] += abs(data[k]-data[i+k])
    return AC


# def autocorr(x):
#     result = np.correlate(x, x, mode='full')
#     return result[result.size/2:]

def autocorr1(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    return r2[:len(x)//2]


Fs, data = read('./Resources/TinHieuMau/lab_male.wav')

altdata = Normalize(data, min(data), max(data))

AC = autocorr1(data)

print(AC)

plt.plot(AC)
plt.show()
