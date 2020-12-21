import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.fftpack import fft,ifft,fftfreq
import math


Fs, data = read('./Resources/TinHieuMau/lab_female.wav')


test = []
for i in range(55125,55426):
    test.append(data[i])

ACtest = abs(fft(test))
# ACtest = ACtest[ACtest.size//2:]
freqs = fftfreq(len(ACtest))*Fs


# for i in range(len(ACtest)):
#     f.append((Fs*i/2 - 1)/len(ACtest))

# for i in range(len(data)):#     for k in range(len(data) - 1):
#         temp += abs(data[k] - data[k - i])
#     D.append(temp)


plt.subplot(2,1,1)
plt.plot(ACtest)

plt.subplot(2,1,2)
plt.plot(data)
plt.show()