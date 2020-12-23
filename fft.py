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


test = []
for i in range(56125,56125+481):
    test.append(data[i])

hamming = np.hamming(480)
ACtest = np.convolve(hamming,test)
ACtest = abs(fft(ACtest,n=480))
# ACtest = ACtest[ACtest.size//2:]

x = []
for i in range(480):
    x.append(i*Fs/480)
peaks, _ = find_peaks(ACtest,height=6000)

# for i in range(len(ACtest)):
#     f.append((Fs*i/2 - 1)/len(ACtest))

# for i in range(len(data)):#     for k in range(len(data) - 1):
#         temp += abs(data[k] - data[k - i])
#     D.append(temp)
print(np.mean(peaks))


plt.plot(x,ACtest)
plt.stem(peaks,ACtest[peaks])


plt.show()