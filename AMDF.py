import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
import scipy.stats.stats as st


def AMDF(arra, arrb):
    cor = np.zeros(len(arra))
    sum = 0
    for i in range(len(arra)):
        position = abs(round((len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += abs(arra[j] - arrb[j+position])
        cor[i] = sum
        sum = 0
    cor = cor[cor.size//2:]
    return cor

Fs, data = read('./Resources/TinHieuMau/lab_male.wav')

altdata = []
for i in range(108694,109932):
    altdata.append(data[i])



amdf = AMDF(altdata,altdata)


plt.plot(amdf)
plt.show()