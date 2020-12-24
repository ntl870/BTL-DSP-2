import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt,hamming
from scipy.fftpack import fft,ifft,fftfreq
import math
from fractions import Fraction as frac

Fs, data = read('./Resources/TinHieuMau/lab_female.wav')

def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res

test = []
for i in range(57125,57125+481):
    test.append(data[i])

hamming = np.hamming(480)
ACtest = np.convolve(hamming,test)
ACtest = fft(ACtest, n=32768)
ACtest = ACtest[:ACtest.size//2]
ACtest = Normalize(ACtest,min(ACtest),max(ACtest))
peaks, _ = find_peaks(ACtest,height=100000)

f = []
for i in range(len(ACtest)):
    f.append((Fs*i/2 - 1)/len(ACtest))

# for i in range(len(data)):#     for k in range(len(data) - 1):
#         temp += abs(data[k] - data[k - i])
#     D.append(temp)
print(np.mean(peaks))


plt.plot(f,ACtest)
# plt.stem(peaks,ACtest[peaks])


plt.show()